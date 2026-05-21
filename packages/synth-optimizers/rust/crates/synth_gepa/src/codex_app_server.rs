use std::collections::{BTreeMap, BTreeSet, VecDeque};
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

const GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION: &str = "gepa_reflective_frame.v1";
const CONTAINER_SENSOR_ADAPTER_ID: &str = "synth.container_sensor_frame_adapter";
const CONTAINER_SENSOR_ADAPTER_VERSION: &str = "v1";
const GEPA_ADAPTER_SOURCE: &str = "https://gepa-ai.github.io/gepa/guides/adapters/";
const GEPA_ALGORITHM_ID: &str = "synth_gepa.v1";

pub(crate) struct CodexProposerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub parent: &'a CandidateRecord,
    pub candidates: &'a [CandidateRecord],
    pub generation: usize,
    pub seed_pool_rows: Value,
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
    let response = json!({
        "backend": "codex_app_server",
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
    });
    write_agent_artifacts(
        input,
        model,
        &thread_id,
        &turn_id,
        &thread_response,
        &final_turn,
        &response,
        client,
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(response)
}

fn materialize_workspace(input: &CodexProposerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let proposal_dir = input.workspace_dir.join("proposal");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&proposal_dir)
        .map_err(|source| OptimizerError::io(&proposal_dir, source))?;

    write_text(
        &input.workspace_dir.join("README.md"),
        &workspace_readme(input),
    )?;
    write_text(
        &proposal_dir.join("PROPOSAL_SCHEMA.md"),
        &proposal_schema(input),
    )?;
    write_json(
        &proposal_dir.join("manifest.json"),
        &json!({
            "schema_version": "gepa_workspace_proposal_v3",
            "critique": "",
            "evidence": {
                "reviewed_files": [],
                "candidate_comparison": "",
                "failure_patterns": [],
                "winning_patterns": [],
                "example_ids_used": [],
            },
            "rationale": "",
            "proposals": [],
        }),
    )?;
    let parent_payload = json!(&input.parent.payload);
    let proposal_request = proposal_request(input);
    let candidates = candidates_read_model(input);
    let candidate_deltas = candidate_deltas_read_model(input);
    let rollouts = rollouts_read_model(input);
    let scores = scores_read_model(input);
    let evidence_frames = evidence_frames_read_model(input);
    let reflective_frames = reflective_frames_read_model(input);
    let links = links_read_model(input);
    let pareto_front = pareto_front_read_model(input);
    let gepa_summary = gepa_summary_read_model(input, &rollouts);
    let candidate_selector = candidate_selector_read_model(input);
    let batch_sampler = batch_sampler_read_model(input);
    let acceptance = acceptance_read_model(input);
    let seed_pools = seed_pools_read_model(input);
    let algorithm_read_model = json!({
        "schema_version": "gepa_algorithm_read_model_v1",
        "generation": input.generation,
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "proposals_per_round": input.config.gepa.proposals_per_generation,
        "candidate_selector": candidate_selector,
        "batch_sampler": batch_sampler,
        "acceptance": acceptance.clone(),
        "seed_pools": seed_pools,
        "reflection_examples": reflection_examples_read_model(input),
        "parent_payload": parent_payload,
        "candidates": candidates,
        "candidate_deltas": candidate_deltas,
        "rollouts": rollouts,
        "scores": scores,
        "evidence_frames": evidence_frames,
        "reflective_frames": reflective_frames,
        "links": links,
        "pareto_front": pareto_front,
        "proposal_request": proposal_request,
        "summary": gepa_summary,
    });
    write_json(
        &state_dir.join("run_context.json"),
        &json!({
            "run_id": input.config.run.run_id,
            "generation": input.generation,
            "task": "GEPA prompt proposal",
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "proposals_per_generation": input.config.gepa.proposals_per_generation,
            "proposals_per_round": input.config.gepa.proposals_per_generation,
            "parent_candidate_id": input.parent.candidate_id,
            "acceptance": acceptance,
            "seed_pool_counts": seed_pool_counts(input),
        }),
    )?;
    write_json(
        &state_dir.join("program_contract.json"),
        &json!({
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "mutable_fields": input.program.mutable_field_ids(),
            "program": input.program,
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
    write_json(&state_dir.join("parent_payload.json"), &parent_payload)?;
    write_json(
        &state_dir.join("candidates.json"),
        &candidates_read_model(input),
    )?;
    write_json(
        &state_dir.join("candidate_deltas.json"),
        &candidate_deltas_read_model(input),
    )?;
    write_json(
        &state_dir.join("rollouts.json"),
        &rollouts_read_model(input),
    )?;
    write_json(&state_dir.join("scores.json"), &scores_read_model(input))?;
    write_json(
        &state_dir.join("evidence_frames.json"),
        &evidence_frames_read_model(input),
    )?;
    write_json(
        &state_dir.join("reflective_frames.json"),
        &reflective_frames_read_model(input),
    )?;
    write_json(&state_dir.join("links.json"), &links_read_model(input))?;
    write_json(
        &state_dir.join("seed_pools.json"),
        &seed_pools_read_model(input),
    )?;
    write_json(
        &state_dir.join("algorithm_read_model.json"),
        &algorithm_read_model,
    )?;
    write_json(
        &state_dir.join("pareto_front.json"),
        &pareto_front_read_model(input),
    )?;
    write_json(&state_dir.join("gepa_sidecar.json"), &algorithm_read_model)?;
    write_json(&state_dir.join("gepa_summary.json"), &gepa_summary)?;
    write_json(&state_dir.join("proposal_request.json"), &proposal_request)?;
    write_json(
        &state_dir.join("reflector_input.json"),
        &reflector_input_read_model(input),
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(())
}

fn write_workspace_pack_manifest(workspace_dir: &Path) -> Result<()> {
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    let mut files = Vec::new();
    collect_workspace_files(workspace_dir, workspace_dir, &mut files)?;
    files.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .cmp(&right.get("path").and_then(Value::as_str))
    });
    write_json(
        &state_dir.join("workspace_pack_manifest.json"),
        &json!({
            "schema_version": "gepa_workspace_pack_manifest.v1",
            "file_count": files.len(),
            "files": files,
        }),
    )
}

fn collect_workspace_files(root: &Path, current: &Path, files: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| OptimizerError::io(current, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(current, source))?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip_workspace_manifest_path(relative) {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.is_dir() {
            collect_workspace_files(root, &path, files)?;
        } else if metadata.is_file() {
            files.push(json!({
                "path": relative.to_string_lossy(),
                "bytes": metadata.len(),
            }));
        }
    }
    Ok(())
}

fn should_skip_workspace_manifest_path(path: &Path) -> bool {
    path.components().any(|component| {
        let text = component.as_os_str().to_string_lossy();
        matches!(text.as_ref(), ".codex_home")
    })
}

#[allow(clippy::too_many_arguments)]
fn write_agent_artifacts(
    input: &CodexProposerInput<'_>,
    model: &str,
    thread_id: &str,
    turn_id: &str,
    thread_response: &Value,
    final_turn: &Value,
    response: &Value,
    client: &AppServerClient,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("opencode_session.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_session.v1",
            "backend": "codex_app_server",
            "model": model,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "workspace": input.workspace_dir,
            "sandbox_mode": input.config.proposer.sandbox_mode,
            "approval_policy": input.config.proposer.approval_policy,
            "thread_response": thread_response,
            "final_turn": final_turn,
        }),
    )?;
    write_json(
        &artifact_dir.join("opencode_messages.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_messages.v1",
            "sent": client.sent_messages,
            "received": client.received_messages,
        }),
    )?;
    write_json(&artifact_dir.join("opencode_response.json"), response)?;
    let mut events = String::new();
    for message in &client.received_messages {
        events.push_str(&serde_json::to_string(message)?);
        events.push('\n');
    }
    write_text(&artifact_dir.join("opencode_sse_events.jsonl"), &events)?;
    Ok(())
}

fn workspace_readme(input: &CodexProposerInput<'_>) -> String {
    format!(
        r#"# GEPA Proposer Workspace

You are proposing the next GEPA prompt candidate.

Read:

1. `proposal/PROPOSAL_SCHEMA.md` for the exact manifest schema.
2. `state/run_context.json` for the optimizer run context and target modules.
3. `state/program_contract.json` for the program and mutable fields.
4. `state/candidates.json` for candidate payloads and train/minibatch/heldout scores.
5. `state/candidate_deltas.json` for payload differences from the selected parent.
6. `state/rollouts.json` and `state/scores.json` for per-example rewards and score summaries.
7. `state/evidence_frames.json`, `state/reflective_frames.json`, and `state/links.json` for durable rollout evidence.
8. `state/seed_pools.json` for pareto-eval, minibatch, reflection, and validation row pools.
9. `state/algorithm_read_model.json` for the complete GEPA read model.
10. `state/pareto_front.json`, `state/gepa_sidecar.json`, and `state/gepa_summary.json` for GEPA-specific mirrors.
11. `state/parent_payload.json` and `state/reflector_input.json` for the parent prompt and sampled wins/losses.

Before writing the manifest, inspect those files with shell, Python, or JQ and form a short evidence summary.
Use a real review workflow: summarize candidate scores and payloads, inspect Pareto membership, inspect rollout wins/losses, inspect the parent payload, then write `proposal/manifest.json`.

Write exactly {proposal_count} distinct candidate proposals to `proposal/manifest.json`.
"#,
        proposal_count = input.config.gepa.proposals_per_generation
    )
}

fn proposal_schema(input: &CodexProposerInput<'_>) -> String {
    let payload_rule = if input.config.candidate.target_modules.len() > 1 {
        "- Multiple target modules are active. Each `proposed_payload` must be the full candidate payload, with every target module present and non-empty.\n"
    } else {
        "- Keep the change targeted to the target module named in `state/run_context.json`.\n"
    };
    format!(
        r#"# GEPA Workspace Proposer Schema

Write `proposal/manifest.json` as strict JSON using this schema:

```json
{{
  "schema_version": "gepa_workspace_proposal_v3",
  "critique": "What the parent candidate is missing, grounded in state/ evidence.",
  "evidence": {{
    "reviewed_files": [
      "state/run_context.json",
      "state/program_contract.json",
      "state/algorithm_read_model.json",
      "state/candidates.json",
      "state/candidate_deltas.json",
      "state/rollouts.json",
      "state/scores.json",
      "state/evidence_frames.json",
      "state/reflective_frames.json",
      "state/seed_pools.json",
      "state/links.json"
    ],
    "candidate_comparison": "Short comparison of parent, Pareto members, and recent candidates.",
    "failure_patterns": ["Observed failure pattern grounded in losing rollout examples."],
    "winning_patterns": ["Observed winning pattern grounded in successful rollout examples."],
    "example_ids_used": ["train:1", "train:10", "train:14"]
  }},
  "rationale": "Why the proposed prompts should improve the target module.",
  "proposals": [
    {{
      "proposal_type": "frontier_variation",
      "parent_candidate_ids": ["<pareto_candidate_id>"],
      "rationale": "Why this variation should help.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }},
    {{
      "proposal_type": "frontier_merge",
      "parent_candidate_ids": ["<pareto_candidate_id_1>", "<pareto_candidate_id_2>"],
      "rationale": "Which strengths this merge attempts to combine.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }}
  ]
}}
```

Rules:

- Read `state/run_context.json`, `state/program_contract.json`, `state/algorithm_read_model.json`, `state/candidates.json`, `state/candidate_deltas.json`, `state/rollouts.json`, `state/scores.json`, `state/evidence_frames.json`, `state/reflective_frames.json`, `state/links.json`, `state/parent_payload.json`, and `state/reflector_input.json`.
- Use shell/Python/JQ inspection to summarize the workspace before writing the manifest. Do not jump straight to editing `proposal/manifest.json`.
- Minimum review workflow: inspect candidate scores/payloads, inspect Pareto membership, inspect rollout wins/losses, inspect parent payload, then write the manifest.
- Use rollout rewards, candidate deltas, reflective frames, failures, wins, and Pareto membership as evidence.
- Fill `evidence` with concrete files reviewed, candidate comparison, failure patterns, winning patterns, and example ids from `state/rollouts.json`.
- Create exactly `state/proposal_request.json.proposals_per_round` distinct proposals.
- Use `proposal_type="frontier_variation"` for a mutation of one Pareto-front candidate.
- Use `proposal_type="frontier_merge"` for an attempted combination of two Pareto-front candidates with complementary wins. If fewer than two Pareto-front candidates exist, replace requested merges with additional frontier variations.
- Do not propose a duplicate of an existing payload in `state/candidates.json`.
- Preserve all parent payload keys unless a key is intentionally changed.
- Each `proposed_payload` must be the full payload object to register as a GEPA candidate.
- For each proposal, at least one targeted module must change from the selected parent payload.
{payload_rule}"#
    )
}

fn proposal_request(input: &CodexProposerInput<'_>) -> Value {
    let proposal_count = input.config.gepa.proposals_per_generation;
    let pareto_front = compute_pareto_front(input);
    let members = sorted_pareto_member_ids(input, &pareto_front);
    let merge_count = if members.len() >= 2 {
        proposal_count / 3
    } else {
        0
    };
    let merge_pairs = merge_candidate_pairs(&members);
    let merge_common_ancestors = merge_pairs
        .iter()
        .map(|pair| {
            (
                pair.join("+"),
                common_ancestor_id(input, &[pair[0].clone(), pair[1].clone()]),
            )
        })
        .collect::<BTreeMap<_, _>>();
    json!({
        "proposal_count": proposal_count,
        "proposals_per_round": proposal_count,
        "frontier_variations": proposal_count.saturating_sub(merge_count),
        "frontier_merges": merge_count,
        "variation_parent_candidate_ids": members,
        "merge_candidate_pairs": merge_pairs,
        "merge_common_ancestors": merge_common_ancestors,
        "frontier_cells": pareto_front.cells.iter().take(200).cloned().collect::<Vec<_>>(),
        "frontier_type": pareto_front.frontier_type,
        "target_modules": input.config.candidate.target_modules,
        "parent_candidate_id": input.parent.candidate_id,
        "candidate_selector": candidate_selector_read_model(input),
        "batch_sampler": batch_sampler_read_model(input),
        "acceptance": acceptance_read_model(input),
        "seed_pool_counts": seed_pool_counts(input),
        "instructions": "Create exactly proposals_per_round distinct candidates. Use frontier_variation for one Pareto-front parent and frontier_merge to combine two complementary Pareto-front parents from merge_candidate_pairs. If no merge pairs are available, replace requested merges with additional frontier variations.",
    })
}

fn sorted_pareto_member_ids(
    input: &CodexProposerInput<'_>,
    front: &CodexParetoFront,
) -> Vec<String> {
    let mut members = front.members.iter().cloned().collect::<Vec<_>>();
    if members.is_empty() {
        members = input
            .candidates
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect();
    }
    members.sort_by(|left, right| {
        let left_wins = front.win_counts.get(left).copied().unwrap_or(0);
        let right_wins = front.win_counts.get(right).copied().unwrap_or(0);
        right_wins.cmp(&left_wins).then_with(|| left.cmp(right))
    });
    members
}

fn merge_candidate_pairs(members: &[String]) -> Vec<Vec<String>> {
    let mut pairs = Vec::new();
    for (left_index, left) in members.iter().enumerate() {
        for right in members.iter().skip(left_index + 1) {
            pairs.push(vec![left.clone(), right.clone()]);
        }
    }
    pairs
}

fn common_ancestor_id(input: &CodexProposerInput<'_>, candidate_ids: &[String]) -> String {
    let Some(first) = candidate_ids.first() else {
        return String::new();
    };
    let chains = candidate_ids
        .iter()
        .map(|candidate_id| ancestor_chain(input, candidate_id))
        .collect::<Vec<_>>();
    for candidate_id in ancestor_chain(input, first) {
        if chains
            .iter()
            .all(|chain| chain.iter().any(|item| item == &candidate_id))
        {
            return candidate_id;
        }
    }
    first.clone()
}

fn ancestor_chain(input: &CodexProposerInput<'_>, candidate_id: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = candidate_id.to_string();
    while seen.insert(current.clone()) {
        chain.push(current.clone());
        let Some(parent_id) = input
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == current)
            .and_then(|candidate| candidate.parent_id.clone())
        else {
            break;
        };
        current = parent_id;
    }
    chain
}

fn candidate_selector_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "name": normalize_candidate_selector_name(&input.config.gepa.candidate_selector.name),
        "configured_name": input.config.gepa.candidate_selector.name,
        "epsilon": input.config.gepa.candidate_selector.epsilon,
        "k": input.config.gepa.candidate_selector.k,
        "frontier_type": normalize_frontier_type(&input.config.gepa.frontier_type),
        "selection_objective": configured_selection_objective(input),
    })
}

fn batch_sampler_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "name": normalize_batch_sampler_name(&input.config.gepa.batch_sampler.name),
        "configured_name": input.config.gepa.batch_sampler.name,
        "epoch_width": input.config.gepa.batch_sampler.epoch_width,
        "field": input.config.gepa.batch_sampler.field,
        "minibatch_size": input.config.gepa.minibatch_size,
        "proposals_per_round": input.config.gepa.proposals_per_generation,
        "objective_keys": input.config.gepa.objective_keys,
        "objective_directions": input.config.gepa.objective_directions,
    })
}

fn acceptance_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "acceptance_criterion": normalize_acceptance_criterion(&input.config.gepa.acceptance_criterion),
        "configured_acceptance_criterion": input.config.gepa.acceptance_criterion,
        "minibatch_accept_margin": input.config.gepa.minibatch_accept_margin,
        "objective_directions": input.config.gepa.objective_directions,
        "objective_acceptance": {
            "min_objective_delta": input.config.gepa.objective_acceptance.min_objective_delta.unwrap_or(0.05),
            "objective_regression_tolerance": input.config.gepa.objective_acceptance.objective_regression_tolerance.unwrap_or(0.10),
            "protected_objectives": input.config.gepa.objective_acceptance.protected_objectives,
        },
    })
}

fn normalize_acceptance_criterion(criterion: &str) -> String {
    match criterion
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "improvement_or_equal" => "improvement_or_equal".to_string(),
        "primary_or_objective" => "primary_or_objective".to_string(),
        "any_objective_improved" => "any_objective_improved".to_string(),
        "protected_objective_guard" => "protected_objective_guard".to_string(),
        _ => "primary_improvement".to_string(),
    }
}

fn seed_pools_read_model(input: &CodexProposerInput<'_>) -> Value {
    if input.seed_pool_rows.is_null() {
        return json!({});
    }
    input.seed_pool_rows.clone()
}

fn seed_pool_counts(input: &CodexProposerInput<'_>) -> Value {
    let mut counts = Map::new();
    if let Some(pools) = input.seed_pool_rows.as_object() {
        for (name, pool) in pools {
            if name == "schema_version" {
                continue;
            }
            let row_count = pool
                .get("row_count")
                .and_then(Value::as_u64)
                .or_else(|| {
                    pool.get("rows")
                        .and_then(Value::as_array)
                        .map(|rows| rows.len() as u64)
                })
                .unwrap_or(0);
            counts.insert(name.clone(), json!(row_count));
        }
    }
    Value::Object(counts)
}

fn reflection_examples_read_model(input: &CodexProposerInput<'_>) -> Value {
    input
        .seed_pool_rows
        .get("reflection")
        .and_then(|pool| pool.get("rows"))
        .and_then(Value::as_array)
        .map(|rows| Value::Array(rows.iter().take(40).cloned().collect()))
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

fn candidates_read_model(input: &CodexProposerInput<'_>) -> Value {
    let pareto_front = compute_pareto_front(input);
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "source": candidate.source,
                    "status": candidate.status,
                    "is_parent": candidate.candidate_id == input.parent.candidate_id,
                    "is_pareto_front": pareto_front.members.contains(&candidate.candidate_id),
                    "payload": candidate.payload,
                    "minibatch_reward": candidate.minibatch_reward,
                    "train_reward": candidate.train_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "minibatch_rollout_count": candidate.minibatch_scores.len(),
                    "train_rollout_count": candidate.train_scores.len(),
                    "sensor_frame_count": candidate.sensor_frames.len(),
                    "acceptance_score": candidate.acceptance_score,
                    "acceptance_metadata": candidate.acceptance_metadata,
                })
            })
            .collect(),
    )
}

fn candidate_deltas_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                let parent_payload = input
                    .candidates
                    .iter()
                    .find(|parent| {
                        Some(parent.candidate_id.as_str()) == candidate.parent_id.as_deref()
                    })
                    .map(|parent| &parent.payload)
                    .unwrap_or(&input.parent.payload);
                let mut changed_modules = Vec::new();
                let mut module_deltas = Map::new();
                for module_id in &input.config.candidate.target_modules {
                    let before = parent_payload.get(module_id).cloned().unwrap_or_default();
                    let after = candidate
                        .payload
                        .get(module_id)
                        .cloned()
                        .unwrap_or_default();
                    if before != after {
                        changed_modules.push(module_id.clone());
                        module_deltas.insert(
                            module_id.clone(),
                            json!({
                                "before": before,
                                "after": after,
                            }),
                        );
                    }
                }
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "changed_modules": changed_modules,
                    "module_deltas": module_deltas,
                })
            })
            .collect(),
    )
}

fn rollouts_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut rows = Vec::new();
    for candidate in input.candidates {
        for score in &candidate.minibatch_scores {
            rows.push(json!({
                "candidate_id": candidate.candidate_id,
                "evaluation_stage": "candidate_minibatch",
                "example_id": score.example_id,
                "seed": score.seed,
                "reward": score.reward,
            }));
        }
        for score in &candidate.train_scores {
            rows.push(json!({
                "candidate_id": candidate.candidate_id,
                "evaluation_stage": "candidate_full_train",
                "example_id": score.example_id,
                "seed": score.seed,
                "reward": score.reward,
            }));
        }
        for frame in &candidate.sensor_frames {
            rows.push(json!({
                "candidate_id": frame.candidate_id,
                "evaluation_stage": frame.evaluation_stage,
                "example_id": frame.example_id,
                "seed": frame.seed,
                "split": frame.split,
                "reward": frame.reward,
                "status": frame.status,
                "success_status": frame.success_status,
                "failure": frame.failure,
                "actionable_side_info": frame.actionable_side_info,
            }));
        }
    }
    Value::Array(rows)
}

fn scores_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status,
                    "source": candidate.source,
                    "minibatch_reward": candidate.minibatch_reward,
                    "train_reward": candidate.train_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "rollout_counts": {
                        "minibatch": candidate.minibatch_scores.len(),
                        "train": candidate.train_scores.len(),
                        "sensor_frames": candidate.sensor_frames.len(),
                    },
                })
            })
            .collect(),
    )
}

fn evidence_frames_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .flat_map(|candidate| candidate.sensor_frames.iter())
            .map(|frame| serde_json::to_value(frame).unwrap_or(Value::Null))
            .filter(|value| !value.is_null())
            .collect(),
    )
}

fn reflective_frames_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut frames = input
        .candidates
        .iter()
        .flat_map(|candidate| {
            candidate
                .sensor_frames
                .iter()
                .map(move |frame| reflective_frame_value(input, candidate, frame))
        })
        .collect::<Vec<_>>();
    frames.sort_by(|left, right| {
        let left_key = left
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let right_key = right
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        left_key.cmp(right_key)
    });
    if frames.len() > 80 {
        frames.truncate(80);
    }
    json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "adapter": reflective_adapter_spec(),
        "frame_count": frames.len(),
        "frames": frames,
    })
}

fn reflective_frame_value(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let component_id = input
        .config
        .candidate
        .target_modules
        .first()
        .cloned()
        .unwrap_or_else(|| "candidate".to_string());
    let rollout_id = frame.rollout_id.clone().unwrap_or_default();
    let trace_refs = frame
        .trace_digest
        .as_ref()
        .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
        .unwrap_or_default();
    let confidence = reflective_confidence(frame);
    let artifact_refs = frame
        .artifact_refs
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    let failure_class = frame
        .failure
        .as_ref()
        .map(|failure| failure.failure_class().to_string())
        .unwrap_or_default();
    let verifier_rationale = frame
        .objective_scores
        .iter()
        .filter_map(|score| score.rationale.as_deref())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
        .to_string();
    let rollout_trace = frame
        .metadata
        .get("rollout_trace")
        .and_then(Value::as_object);
    let trace_summary = rollout_trace
        .and_then(|trace| trace.get("summary"))
        .cloned()
        .or_else(|| frame.metadata.get("summary").cloned())
        .unwrap_or(Value::Null);
    let trace_outcome = rollout_trace
        .and_then(|trace| trace.get("outcome"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "status": frame.status,
                "success_status": frame.success_status,
                "reward": frame.reward,
            })
        });
    let task_example = rollout_trace
        .and_then(|trace| trace.get("task_payload"))
        .and_then(|task_payload| task_payload.get("example"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "example_id": frame.example_id,
                "seed": frame.seed,
                "split": frame.split,
            })
        });
    let request = rollout_trace
        .and_then(|trace| trace.get("request"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "evaluation_stage": frame.evaluation_stage,
                "target_modules": input.config.candidate.target_modules,
            })
        });
    let tool_calls = rollout_trace
        .and_then(|trace| trace.get("tool_calls"))
        .cloned()
        .unwrap_or_else(|| json!([]));
    let substitution_stats = rollout_trace
        .and_then(|trace| trace.get("substitution_stats"))
        .cloned()
        .unwrap_or_else(|| json!({"attempted": 0, "applied": 0, "warnings": []}));
    let evidence = json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "source": "sensor_frame_adapter",
        "adapter": reflective_adapter_spec(),
        "subject": {
            "algorithm_id": GEPA_ALGORITHM_ID,
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_id,
            "component_id": component_id,
            "rollout_id": rollout_id,
            "example_id": frame.example_id,
        },
        "adapter_source": GEPA_ADAPTER_SOURCE,
        "rollout_id": rollout_id,
        "example_id": frame.example_id,
        "split": frame.split,
        "inputs": {
            "example": task_example,
            "request": request,
        },
        "generated_outputs": {
            "summary": trace_summary,
            "outcome": trace_outcome,
        },
        "feedback": {
            "reward": frame.reward,
            "objective_scores": frame.objective_scores,
            "verifier_rationale": verifier_rationale,
        },
        "actionable_side_info": frame.actionable_side_info.clone().unwrap_or_else(|| json!({})),
        "sensors": {
            "confidence": confidence,
            "trace_digest": frame.trace_digest,
        },
        "refs": {
            "trace_refs": trace_refs,
            "rollout_id": rollout_id,
            "sensor_frame_id": frame.sensor_frame_id,
            "artifact_refs": artifact_refs,
        },
        "trace_refs": trace_refs,
        "tool_calls": tool_calls,
        "substitution_stats": substitution_stats,
        "failure_class": failure_class,
        "usage": frame.usage,
        "confidence": confidence,
        "component_id": component_id,
    });
    json!({
        "frame_id": format!("reflect:{}:{}:{}", GEPA_ALGORITHM_ID, candidate.candidate_id, frame.sensor_frame_id),
        "algorithm_id": GEPA_ALGORITHM_ID,
        "component_id": component_id,
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "rollout_id": frame.rollout_id,
        "artifact_refs": artifact_refs,
        "metadata": {
            "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
            "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
            "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
            "sensor_frame_id": frame.sensor_frame_id,
        },
        "evidence": evidence,
    })
}

fn reflective_adapter_spec() -> Value {
    json!({
        "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
        "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
        "source": GEPA_ADAPTER_SOURCE,
        "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "required_evidence_keys": [
            "schema_version",
            "source",
            "adapter",
            "subject",
            "inputs",
            "generated_outputs",
            "feedback",
            "actionable_side_info",
            "sensors",
            "refs",
        ],
    })
}

fn reflective_confidence(frame: &synth_optimizer_platform::SensorFrame) -> f64 {
    let support_count = frame
        .trace_digest
        .as_ref()
        .map(|digest| digest.llm_request_count + digest.tool_call_count)
        .unwrap_or(0);
    if frame.failure.is_some() {
        0.55
    } else if support_count > 0 {
        0.85
    } else if frame.actionable_side_info.is_some() {
        0.7
    } else {
        0.35
    }
}

fn links_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut links = Vec::new();
    for candidate in input.candidates {
        if let Some(parent_id) = &candidate.parent_id {
            links.push(json!({
                "type": "candidate_parent",
                "from": candidate.candidate_id,
                "to": parent_id,
            }));
        }
        for frame in &candidate.sensor_frames {
            links.push(json!({
                "type": "candidate_rollout_evidence",
                "from": candidate.candidate_id,
                "to": frame.sensor_frame_id,
                "example_id": frame.example_id,
                "evaluation_stage": frame.evaluation_stage,
            }));
        }
    }
    Value::Array(links)
}

fn pareto_front_read_model(input: &CodexProposerInput<'_>) -> Value {
    let pareto_front = compute_pareto_front(input);
    let mut members = pareto_front
        .members
        .iter()
        .filter_map(|candidate_id| {
            input
                .candidates
                .iter()
                .find(|candidate| &candidate.candidate_id == candidate_id)
                .map(|candidate| {
                    let win_count = pareto_front
                        .win_counts
                        .get(&candidate.candidate_id)
                        .copied()
                        .unwrap_or(0);
                    json!({
                        "candidate_id": candidate.candidate_id,
                        "parent_id": candidate.parent_id,
                        "source": candidate.source,
                        "status": candidate.status,
                        "train_reward": candidate.train_reward,
                        "minibatch_reward": candidate.minibatch_reward,
                        "heldout_reward": candidate.heldout_reward,
                        "win_count": win_count,
                        "payload": candidate.payload,
                    })
                })
        })
        .collect::<Vec<_>>();
    members.sort_by(|left, right| {
        left.get("candidate_id")
            .and_then(Value::as_str)
            .cmp(&right.get("candidate_id").and_then(Value::as_str))
    });
    json!({
        "schema_version": "gepa_pareto_front.v1",
        "frontier_type": pareto_front.frontier_type,
        "score_source": pareto_front.score_source,
        "objective_keys": input.config.gepa.objective_keys,
        "objective_directions": input.config.gepa.objective_directions,
        "parent_candidate_id": input.parent.candidate_id,
        "candidate_selector": candidate_selector_read_model(input),
        "members": members,
        "win_counts": pareto_front.win_counts,
        "cells": pareto_front.cells,
        "legacy_status_frontier": legacy_frontier_read_model(input),
    })
}

#[derive(Debug)]
struct CodexParetoFront {
    frontier_type: String,
    score_source: String,
    members: BTreeSet<String>,
    win_counts: BTreeMap<String, usize>,
    cells: Vec<Value>,
}

fn compute_pareto_front(input: &CodexProposerInput<'_>) -> CodexParetoFront {
    let frontier_type = normalize_frontier_type(&input.config.gepa.frontier_type);
    let mut cells = match frontier_type.as_str() {
        "per_objective" => codex_pareto_objective_cells(input),
        "per_example_objective" => codex_pareto_example_objective_cells(input),
        _ => codex_pareto_example_cells(input),
    };
    if cells.is_empty() && frontier_type != "per_example" {
        cells = codex_pareto_example_cells(input);
    }
    let mut members = BTreeSet::new();
    let mut win_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut cell_values = Vec::new();
    for cell in cells {
        members.insert(cell.candidate_id.clone());
        *win_counts.entry(cell.candidate_id.clone()).or_default() += 1;
        cell_values.push(json!({
            "frontier_key": cell.frontier_key,
            "candidate_id": cell.candidate_id,
            "score": cell.score,
            "example_id": cell.example_id,
            "objective_id": cell.objective_id,
        }));
    }
    if members.is_empty() {
        for candidate in input.candidates {
            if candidate.train_reward.is_some() {
                members.insert(candidate.candidate_id.clone());
                win_counts.insert(candidate.candidate_id.clone(), 1);
                cell_values.push(json!({
                    "frontier_key": format!("candidate:{}", candidate.candidate_id),
                    "candidate_id": candidate.candidate_id,
                    "score": candidate.train_reward,
                    "example_id": Value::Null,
                    "objective_id": Value::Null,
                }));
            }
        }
    }
    CodexParetoFront {
        frontier_type,
        score_source: "sensor_frame.objective_scores".to_string(),
        members,
        win_counts,
        cells: cell_values,
    }
}

#[derive(Clone, Debug)]
struct CodexParetoCell {
    frontier_key: String,
    candidate_id: String,
    score: f64,
    example_id: Option<String>,
    objective_id: Option<String>,
}

fn codex_pareto_example_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let selection_objective = configured_selection_objective(input);
    let selection_direction = selection_objective
        .as_deref()
        .map(|objective| codex_objective_direction(input, objective))
        .unwrap_or(1.0);
    let mut winners: BTreeMap<String, CodexParetoCell> = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            let candidate_id = candidate.candidate_id.clone();
            let score = frame_objective_score(frame, selection_objective.as_deref())
                .unwrap_or(frame.reward);
            upsert_codex_pareto_cell(
                &mut winners,
                frame.example_id.clone(),
                CodexParetoCell {
                    frontier_key: format!("example:{}", frame.example_id),
                    candidate_id,
                    score: score * selection_direction,
                    example_id: Some(frame.example_id.clone()),
                    objective_id: None,
                },
            );
        }
    }
    winners.into_values().collect()
}

fn codex_pareto_objective_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let objective_keys = configured_objective_keys(input);
    let mut sums: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            for score in &frame.objective_scores {
                if !objective_keys.is_empty() && !objective_keys.contains(&score.objective) {
                    continue;
                }
                let entry = sums
                    .entry((candidate.candidate_id.clone(), score.objective.clone()))
                    .or_insert((0.0, 0));
                entry.0 += score.value;
                entry.1 += 1;
            }
        }
    }
    let mut winners = BTreeMap::new();
    for ((candidate_id, objective), (sum, count)) in sums {
        if count == 0 {
            continue;
        }
        upsert_codex_pareto_cell(
            &mut winners,
            objective.clone(),
            CodexParetoCell {
                frontier_key: format!("objective:{objective}"),
                candidate_id,
                score: (sum / count as f64) * codex_objective_direction(input, &objective),
                example_id: None,
                objective_id: Some(objective),
            },
        );
    }
    winners.into_values().collect()
}

fn codex_pareto_example_objective_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let objective_keys = configured_objective_keys(input);
    let mut winners = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            for score in &frame.objective_scores {
                if !objective_keys.is_empty() && !objective_keys.contains(&score.objective) {
                    continue;
                }
                let key = format!("{}|{}", frame.example_id, score.objective);
                upsert_codex_pareto_cell(
                    &mut winners,
                    key,
                    CodexParetoCell {
                        frontier_key: format!(
                            "example_objective:{}|{}",
                            frame.example_id, score.objective
                        ),
                        candidate_id: candidate.candidate_id.clone(),
                        score: score.value * codex_objective_direction(input, &score.objective),
                        example_id: Some(frame.example_id.clone()),
                        objective_id: Some(score.objective.clone()),
                    },
                );
            }
        }
    }
    winners.into_values().collect()
}

fn train_sensor_frames(
    candidate: &CandidateRecord,
) -> impl Iterator<Item = &synth_optimizer_platform::SensorFrame> {
    candidate.sensor_frames.iter().filter(|frame| {
        matches!(
            frame.evaluation_stage.as_str(),
            "seed_full_train" | "candidate_full_train"
        )
    })
}

fn upsert_codex_pareto_cell(
    winners: &mut BTreeMap<String, CodexParetoCell>,
    key: String,
    challenger: CodexParetoCell,
) {
    let should_replace = winners
        .get(&key)
        .map(|incumbent| {
            challenger.score > incumbent.score + f64::EPSILON
                || ((challenger.score - incumbent.score).abs() <= f64::EPSILON
                    && challenger.candidate_id < incumbent.candidate_id)
        })
        .unwrap_or(true);
    if should_replace {
        winners.insert(key, challenger);
    }
}

fn configured_selection_objective(input: &CodexProposerInput<'_>) -> Option<String> {
    input
        .config
        .gepa
        .selection_objective
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn configured_objective_keys(input: &CodexProposerInput<'_>) -> BTreeSet<String> {
    input
        .config
        .gepa
        .objective_keys
        .iter()
        .map(|objective| objective.trim())
        .filter(|objective| !objective.is_empty())
        .map(str::to_string)
        .collect()
}

fn codex_objective_direction(input: &CodexProposerInput<'_>, objective: &str) -> f64 {
    input
        .config
        .gepa
        .objective_directions
        .get(objective)
        .map(String::as_str)
        .map(normalize_objective_direction)
        .unwrap_or(1.0)
}

fn normalize_objective_direction(direction: &str) -> f64 {
    match direction.trim().to_ascii_lowercase().as_str() {
        "min" | "minimize" | "lower" | "lower_is_better" | "down" => -1.0,
        _ => 1.0,
    }
}

fn frame_objective_score(
    frame: &synth_optimizer_platform::SensorFrame,
    objective: Option<&str>,
) -> Option<f64> {
    let objective = objective?;
    frame
        .objective_scores
        .iter()
        .find(|score| score.objective == objective)
        .map(|score| score.value)
}

fn normalize_frontier_type(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "per_objective" => "per_objective".to_string(),
        "per_example_objective" => "per_example_objective".to_string(),
        _ => "per_example".to_string(),
    }
}

fn normalize_candidate_selector_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "pareto" | "pareto_weighted" => "pareto_weighted".to_string(),
        "uniform_pareto" => "uniform_pareto".to_string(),
        "random" => "random".to_string(),
        "current_best" => "current_best".to_string(),
        "top_k_pareto" => "top_k_pareto".to_string(),
        "epsilon_greedy" => "epsilon_greedy".to_string(),
        _ => "pareto_weighted".to_string(),
    }
}

fn normalize_batch_sampler_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "epoch_shuffled" => "epoch_shuffled".to_string(),
        "ordered_epoch" | "sequential_epoch" => "ordered_epoch".to_string(),
        "stratified" | "stratified_by_field" => "stratified".to_string(),
        _ => "seeded_shuffle".to_string(),
    }
}

fn gepa_summary_read_model(input: &CodexProposerInput<'_>, rollouts: &Value) -> Value {
    let pareto_front = compute_pareto_front(input);
    let best = input.candidates.iter().max_by(|left, right| {
        score_for_order(left)
            .partial_cmp(&score_for_order(right))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let example_ids = rollouts
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("example_id").and_then(Value::as_str))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    json!({
        "candidate_count": input.candidates.len(),
        "frontier_count": pareto_front.members.len(),
        "frontier_type": pareto_front.frontier_type,
        "candidate_selector": candidate_selector_read_model(input),
        "batch_sampler": batch_sampler_read_model(input),
        "parent_candidate_id": input.parent.candidate_id,
        "best_candidate_id": best.map(|candidate| candidate.candidate_id.as_str()),
        "best_train_reward": best.and_then(|candidate| candidate.train_reward),
        "observed_example_count": example_ids.len(),
        "rollout_row_count": rollouts.as_array().map(Vec::len).unwrap_or(0),
    })
}

fn legacy_frontier_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.status == "accepted"
                    || candidate.status == "seed"
                    || candidate.heldout_reward.is_some()
            })
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "train_reward": candidate.train_reward,
                    "minibatch_reward": candidate.minibatch_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "payload": candidate.payload,
                })
            })
            .collect(),
    )
}

fn reflector_input_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut wins = Vec::new();
    let mut losses = Vec::new();
    for frame in input
        .candidates
        .iter()
        .flat_map(|candidate| candidate.sensor_frames.iter())
    {
        let row = json!({
            "candidate_id": frame.candidate_id,
            "evaluation_stage": frame.evaluation_stage,
            "example_id": frame.example_id,
            "reward": frame.reward,
            "status": frame.status,
            "failure": frame.failure,
            "actionable_side_info": frame.actionable_side_info,
        });
        if frame.reward >= 1.0 {
            wins.push(row);
        } else {
            losses.push(row);
        }
    }
    wins.truncate(20);
    losses.truncate(20);
    json!({
        "parent_candidate_id": input.parent.candidate_id,
        "wins": wins,
        "losses": losses,
    })
}

fn score_for_order(candidate: &CandidateRecord) -> f64 {
    candidate
        .train_reward
        .or(candidate.minibatch_reward)
        .or(candidate.heldout_reward)
        .unwrap_or(f64::NEG_INFINITY)
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
        "Read README.md, proposal/PROPOSAL_SCHEMA.md, and all files under state/.\n\
         Use shell/Python/JQ tools to inspect candidates, Pareto data, and rollout failures before editing proposal/manifest.json.\n\
         Propose exactly {} prompt candidates for generation {}.\n\
         Use only these target modules: {}.\n\
         Write strict JSON to proposal/manifest.json using schema_version gepa_workspace_proposal_v3.\n\
         Include the required evidence block with reviewed files, candidate comparison, failure patterns, winning patterns, and example ids.\n\
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
    sent_messages: Vec<Value>,
    received_messages: Vec<Value>,
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
            sent_messages: Vec::new(),
            received_messages: Vec::new(),
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
        self.sent_messages.push(payload);
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
            Ok(result) => match result {
                Ok(message) => {
                    self.received_messages.push(message.clone());
                    Ok(message)
                }
                Err(error) => Err(error),
            },
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
    validate_manifest_contract(manifest)?;
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

fn validate_manifest_contract(manifest: &Value) -> Result<()> {
    let schema_version = manifest
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema_version != "gepa_workspace_proposal_v3" {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer manifest schema_version={schema_version:?}; expected gepa_workspace_proposal_v3"
        )));
    }
    let evidence = manifest
        .get("evidence")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "codex app-server proposer manifest omitted required evidence object".to_string(),
            )
        })?;
    let reviewed = evidence
        .get("reviewed_files")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    let required = [
        "state/run_context.json",
        "state/program_contract.json",
        "state/algorithm_read_model.json",
        "state/candidates.json",
        "state/candidate_deltas.json",
        "state/rollouts.json",
        "state/scores.json",
        "state/evidence_frames.json",
        "state/links.json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();
    let missing = required.difference(&reviewed).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer evidence missing reviewed_files={missing:?}"
        )));
    }
    for field in [
        "candidate_comparison",
        "failure_patterns",
        "winning_patterns",
        "example_ids_used",
    ] {
        let Some(value) = evidence.get(field) else {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence missing {field}"
            )));
        };
        let has_content = match value {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => items.iter().any(|item| {
                item.as_str()
                    .map(|text| !text.trim().is_empty())
                    .unwrap_or(false)
            }),
            _ => false,
        };
        if !has_content {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence field {field} is empty"
            )));
        }
    }
    Ok(())
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
