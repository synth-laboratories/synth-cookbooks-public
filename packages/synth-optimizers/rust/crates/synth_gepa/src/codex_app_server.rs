use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use synth_optimizer_platform::{OptimizerError, PromptProgram, Result, SynthOptimizerConfig};

use crate::CandidateRecord;

pub(crate) struct CodexProposerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub parent: &'a CandidateRecord,
    pub candidates: &'a [CandidateRecord],
    pub generation: usize,
    pub workspace_dir: PathBuf,
}

pub(crate) fn run_codex_app_server_proposer(input: CodexProposerInput<'_>) -> Result<Value> {
    materialize_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let mut client = AppServerClient::start(&input, &model)?;
    let result = run_session(&mut client, &input, &model);
    let terminate_result = client.terminate();
    let mut response = result?;
    if let Err(error) = terminate_result {
        response["shutdown_warning"] = Value::String(error.to_string());
    }
    Ok(response)
}

fn run_session(
    client: &mut AppServerClient,
    input: &CodexProposerInput<'_>,
    model: &str,
) -> Result<Value> {
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let initialize_id = client.send_request(
        "initialize",
        json!({
            "clientInfo": {
                "name": "synth-optimizers-gepa",
                "title": "synth-optimizers GEPA",
                "version": env!("CARGO_PKG_VERSION"),
            }
        }),
    )?;
    client.wait_for_response(initialize_id, Duration::from_secs(60))?;
    client.send_notification("initialized", Value::Null)?;

    let thread_id = client.send_request("thread/start", thread_start_params(input, model))?;
    let thread_response = client.wait_for_response(thread_id, Duration::from_secs(60))?;
    let thread_id = extract_thread_id(&thread_response).ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "codex app-server thread/start response missing thread id: {thread_response}"
        ))
    })?;

    let turn_id = client.send_request("turn/start", turn_start_params(input, model, &thread_id))?;
    let turn_id = client.wait_for_turn_started(turn_id, Duration::from_secs(60))?;
    let final_turn = client.wait_for_turn(&turn_id, timeout)?;
    ensure_turn_completed(&final_turn)?;

    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let usage = usage_from_message(&final_turn)
        .unwrap_or_else(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}));
    Ok(json!({
        "backend": "codex_app_server",
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
    }))
}

fn materialize_workspace(input: &CodexProposerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let proposal_dir = input.workspace_dir.join("proposal");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&proposal_dir)
        .map_err(|source| OptimizerError::io(&proposal_dir, source))?;

    write_text(
        &input.workspace_dir.join("README.md"),
        &format!(
            r#"# GEPA Proposer Workspace

You are proposing the next GEPA prompt candidate.

Read `proposal/PROPOSAL_SCHEMA.md` and the JSON files under `state/`.
Use the evidence in the state files before editing `proposal/manifest.json`.

Write exactly {proposal_count} proposals. Each proposal must include a full
`proposed_payload` object with all target modules present.
"#,
            proposal_count = input.config.gepa.proposals_per_generation
        ),
    )?;
    write_text(
        &proposal_dir.join("PROPOSAL_SCHEMA.md"),
        r#"# GEPA Workspace Proposal Schema

Write `proposal/manifest.json` as strict JSON:

```json
{
  "schema_version": "gepa_workspace_proposal_v1",
  "critique": "Short diagnosis grounded in the state files.",
  "evidence": {
    "reviewed_files": ["state/run_context.json", "state/program.json", "state/candidates.json"],
    "candidate_comparison": "Short comparison of available candidates.",
    "failure_patterns": ["Pattern the proposal addresses."],
    "winning_patterns": ["Pattern to preserve."],
    "example_ids_used": ["train:0"]
  },
  "proposals": [
    {
      "proposal_type": "frontier_variation",
      "parent_candidate_ids": ["<parent id>"],
      "rationale": "Why this candidate should improve reward.",
      "proposed_payload": {
        "<target module>": "<full replacement instruction>"
      }
    }
  ]
}
```

Rules:

- Do not invent target module ids.
- Preserve existing payload keys unless intentionally changing a target module.
- Do not leave `proposal/manifest.json` empty.
- Return the proposal by writing the manifest file, not only in chat.
"#,
    )?;
    write_json(
        &proposal_dir.join("manifest.json"),
        &json!({
            "schema_version": "gepa_workspace_proposal_v1",
            "critique": "",
            "evidence": {
                "reviewed_files": [],
                "candidate_comparison": "",
                "failure_patterns": [],
                "winning_patterns": [],
                "example_ids_used": [],
            },
            "proposals": [],
        }),
    )?;
    write_json(
        &state_dir.join("run_context.json"),
        &json!({
            "run_id": input.config.run.run_id,
            "generation": input.generation,
            "task": "GEPA prompt proposal",
            "target_modules": input.config.candidate.target_modules,
            "proposals_per_generation": input.config.gepa.proposals_per_generation,
            "parent_candidate_id": input.parent.candidate_id,
        }),
    )?;
    write_json(
        &state_dir.join("program.json"),
        &serde_json::to_value(input.program)?,
    )?;
    write_json(
        &state_dir.join("parent_candidate.json"),
        &serde_json::to_value(input.parent)?,
    )?;
    write_json(
        &state_dir.join("candidates.json"),
        &serde_json::to_value(input.candidates)?,
    )?;
    write_json(
        &state_dir.join("proposal_request.json"),
        &json!({
            "proposal_count": input.config.gepa.proposals_per_generation,
            "target_modules": input.config.candidate.target_modules,
            "parent_candidate_id": input.parent.candidate_id,
        }),
    )?;
    Ok(())
}

fn thread_start_params(input: &CodexProposerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the GEPA workspace proposer. Work only inside this workspace.".to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn turn_start_params(input: &CodexProposerInput<'_>, model: &str, thread_id: &str) -> Value {
    let mut params = Map::new();
    params.insert("threadId".to_string(), Value::String(thread_id.to_string()));
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": proposer_instructions(input),
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Value::Object(params)
}

fn proposer_instructions(input: &CodexProposerInput<'_>) -> String {
    format!(
        "Read README.md, proposal/PROPOSAL_SCHEMA.md, and every JSON file under state/.\n\
         Propose exactly {} prompt candidates for generation {}.\n\
         Use only these target modules: {}.\n\
         Write strict JSON to proposal/manifest.json using schema_version gepa_workspace_proposal_v1.\n\
         Do not print pseudo-tool calls. Use real file inspection and file editing.",
        input.config.gepa.proposals_per_generation,
        input.generation,
        input.config.candidate.target_modules.join(", ")
    )
}

fn sandbox_policy_for_mode(mode: &str) -> Value {
    match mode {
        "danger-full-access" => json!({"type": "dangerFullAccess"}),
        "read-only" => {
            json!({"type": "readOnly", "access": {"type": "fullAccess"}, "networkAccess": true})
        }
        "workspace-write" => {
            json!({"type": "workspaceWrite", "readOnlyAccess": {"type": "fullAccess"}, "networkAccess": true})
        }
        _ => Value::String(mode.to_string()),
    }
}

struct AppServerClient {
    child: Child,
    stdin: ChildStdin,
    receiver: Receiver<Result<Value>>,
    buffer: VecDeque<Value>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    next_id: u64,
}

impl AppServerClient {
    fn start(input: &CodexProposerInput<'_>, model: &str) -> Result<Self> {
        let workspace_dir = fs::canonicalize(&input.workspace_dir)
            .map_err(|source| OptimizerError::io(&input.workspace_dir, source))?;
        let command = if input.config.proposer.command.is_empty() {
            vec!["codex".to_string(), "app-server".to_string()]
        } else {
            input.config.proposer.command.clone()
        };
        let mut env_map = env::vars().collect::<BTreeMap<_, _>>();
        if let Some(api_key_env) = non_empty(input.config.proposer.api_key_env.as_deref()) {
            if let Ok(api_key) = env::var(api_key_env) {
                if !api_key.trim().is_empty() {
                    env_map.insert("OPENAI_API_KEY".to_string(), api_key);
                }
            }
        }
        if input.config.proposer.copy_host_auth {
            let codex_home = workspace_dir.join(".codex_home");
            copy_codex_home(&codex_home)?;
            env_map.insert("CODEX_HOME".to_string(), codex_home.display().to_string());
        }
        let mut cmd = Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(&workspace_dir)
            .envs(env_map)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|source| {
            OptimizerError::Proposer(format!(
                "failed to start codex app-server command {:?} for model {}: {}",
                command, model, source
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            OptimizerError::Proposer("codex app-server stdin unavailable".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            OptimizerError::Proposer("codex app-server stdout unavailable".to_string())
        })?;
        let stderr = child.stderr.take();
        let (sender, receiver) = mpsc::channel();
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        thread::spawn(move || read_stdout(stdout, sender));
        if let Some(stderr) = stderr {
            let stderr_tail = Arc::clone(&stderr_tail);
            thread::spawn(move || drain_stderr(stderr, stderr_tail));
        }
        Ok(Self {
            child,
            stdin,
            receiver,
            buffer: VecDeque::new(),
            stderr_tail,
            next_id: 1,
        })
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok(id)
    }

    fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    fn send(&mut self, payload: Value) -> Result<()> {
        serde_json::to_writer(&mut self.stdin, &payload)?;
        self.stdin
            .write_all(b"\n")
            .map_err(|source| OptimizerError::io("codex app-server stdin", source))?;
        self.stdin
            .flush()
            .map_err(|source| OptimizerError::io("codex app-server stdin", source))
    }

    fn wait_for_response(&mut self, id: u64, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
            if message.get("id").and_then(Value::as_u64) == Some(id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "codex app-server request {id} failed: {error}"
                    )));
                }
                self.restore_deferred(deferred);
                return Ok(message);
            }
            deferred.push(message);
        }
    }

    fn wait_for_turn_started(&mut self, request_id: u64, timeout: Duration) -> Result<String> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
            if message.get("id").and_then(Value::as_u64) == Some(request_id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "codex app-server turn/start request failed: {error}"
                    )));
                }
                let turn_id = extract_turn_id(&message).ok_or_else(|| {
                    OptimizerError::Proposer(format!(
                        "codex app-server turn/start response missing turn id: {message}"
                    ))
                })?;
                self.restore_deferred(deferred);
                return Ok(turn_id);
            }
            if message.get("method").and_then(Value::as_str) == Some("turn/started") {
                if let Some(turn_id) = extract_turn_id(&message) {
                    self.restore_deferred(deferred);
                    return Ok(turn_id);
                }
            }
            deferred.push(message);
        }
    }

    fn restore_deferred(&mut self, deferred: Vec<Value>) {
        for message in deferred.into_iter().rev() {
            self.buffer.push_front(message);
        }
    }

    fn wait_for_turn(&mut self, turn_id: &str, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            let message = self.read_next(deadline)?;
            let method = message
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let matching_turn = match message_turn_id(&message) {
                Some(observed) => observed == turn_id,
                None => true,
            };
            if matches!(
                method,
                "turn/completed" | "turn/failed" | "turn/interrupted"
            ) && matching_turn
            {
                return Ok(message);
            }
        }
    }

    fn read_next(&mut self, deadline: Instant) -> Result<Value> {
        let now = Instant::now();
        if now >= deadline {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server timed out waiting for response{}",
                self.stderr_tail_suffix()
            )));
        }
        if let Some(message) = self.buffer.pop_front() {
            return Ok(message);
        }
        match self.receiver.recv_timeout(deadline - now) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(OptimizerError::Proposer(format!(
                "codex app-server timed out waiting for response{}",
                self.stderr_tail_suffix()
            ))),
            Err(RecvTimeoutError::Disconnected) => Err(OptimizerError::Proposer(format!(
                "codex app-server stdout closed{}",
                self.stderr_tail_suffix()
            ))),
        }
    }

    fn stderr_tail_suffix(&self) -> String {
        let Ok(tail) = self.stderr_tail.lock() else {
            return String::new();
        };
        if tail.is_empty() {
            return String::new();
        }
        format!(
            "; stderr_tail={}",
            tail.iter().cloned().collect::<Vec<_>>().join("").trim()
        )
    }

    fn terminate(&mut self) -> Result<()> {
        if self
            .child
            .try_wait()
            .map_err(|source| {
                OptimizerError::Proposer(format!("failed to inspect codex app-server: {source}"))
            })?
            .is_some()
        {
            return Ok(());
        }
        self.child.kill().map_err(|source| {
            OptimizerError::Proposer(format!("failed to stop codex app-server: {source}"))
        })?;
        let _ = self.child.wait();
        Ok(())
    }
}

fn read_stdout(stdout: impl Read, sender: mpsc::Sender<Result<Value>>) {
    let mut reader = BufReader::new(stdout);
    loop {
        match read_jsonrpc_message(&mut reader) {
            Ok(Some(value)) => {
                if sender.send(Ok(value)).is_err() {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let _ = sender.send(Err(error));
                return;
            }
        }
    }
}

fn drain_stderr(stderr: impl Read, tail: Arc<Mutex<VecDeque<String>>>) {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        if let Ok(mut tail) = tail.lock() {
            if tail.len() >= 50 {
                tail.pop_front();
            }
            tail.push_back(line.clone());
        }
        line.clear();
    }
}

fn read_jsonrpc_message(reader: &mut BufReader<impl Read>) -> Result<Option<Value>> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Ok(Some(serde_json::from_str(trimmed)?));
        }
        let mut headers = BTreeMap::new();
        if let Some((key, value)) = trimmed.split_once(':') {
            headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
        loop {
            line.clear();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
            if bytes == 0 {
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let raw_len = headers.get("content-length").ok_or_else(|| {
            OptimizerError::Proposer("codex app-server message missing Content-Length".to_string())
        })?;
        let len = raw_len.parse::<usize>().map_err(|source| {
            OptimizerError::Proposer(format!(
                "invalid codex app-server Content-Length {raw_len}: {source}"
            ))
        })?;
        let mut payload = vec![0u8; len];
        reader
            .read_exact(&mut payload)
            .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
        return Ok(Some(serde_json::from_slice(&payload)?));
    }
}

fn ensure_turn_completed(message: &Value) -> Result<()> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if method == "turn/completed" {
        let status = message
            .pointer("/params/turn/status")
            .and_then(Value::as_str)
            .unwrap_or("completed");
        if status == "completed" {
            return Ok(());
        }
    }
    Err(OptimizerError::Proposer(format!(
        "codex app-server turn did not complete: {message}"
    )))
}

fn read_manifest(workspace_dir: &Path) -> Result<Value> {
    let path = workspace_dir.join("proposal").join("manifest.json");
    let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
    if text.trim().is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer wrote an empty manifest: {}",
            path.display()
        )));
    }
    Ok(serde_json::from_str(&text)?)
}

fn proposals_from_manifest(manifest: &Value) -> Result<Value> {
    let proposals = manifest
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if proposals.is_empty() {
        return Err(OptimizerError::Proposer(
            "codex app-server proposer manifest contained no proposals".to_string(),
        ));
    }
    Ok(Value::Array(proposals))
}

fn extract_thread_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/thread/id")
        .or_else(|| message.pointer("/result/threadId"))
        .or_else(|| message.pointer("/params/thread/id"))
        .or_else(|| message.pointer("/params/threadId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn extract_turn_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/turn/id")
        .or_else(|| message.pointer("/result/turnId"))
        .or_else(|| message.pointer("/params/turn/id"))
        .or_else(|| message.pointer("/params/turnId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn message_turn_id(message: &Value) -> Option<String> {
    extract_turn_id(message)
}

fn usage_from_message(message: &Value) -> Option<Value> {
    let usage = message
        .pointer("/params/turn/usage")
        .or_else(|| message.pointer("/params/usage"))
        .or_else(|| message.pointer("/result/usage"))?;
    Some(json!({
        "prompt_tokens": usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "completion_tokens": usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0),
    }))
}

fn copy_codex_home(destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    let source = env::var("CODEX_HOME").map(PathBuf::from).ok().or_else(|| {
        env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".codex"))
    });
    let Some(source) = source else {
        return Ok(());
    };
    for filename in [
        "auth.json",
        "installation_id",
        "version.json",
        "models_cache.json",
    ] {
        let source_file = source.join(filename);
        if source_file.is_file() {
            let destination_file = destination.join(filename);
            fs::copy(&source_file, &destination_file)
                .map_err(|copy_error| OptimizerError::io(destination_file, copy_error))?;
        }
    }
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    write_text(path, &format!("{text}\n"))
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).map_err(|source| OptimizerError::io(path, source))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
