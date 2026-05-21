use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::artifacts::ArtifactRef;
use crate::cache::stable_json;
use crate::container_contract::{dataset_row_identity, RolloutResponse};
use crate::error::Result;
use crate::failures::{FailurePayload, OptimizerFailureType};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectiveScore {
    pub objective: String,
    pub value: f64,
    pub source: String,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDigest {
    pub sha256: String,
    pub event_count: usize,
    pub llm_request_count: usize,
    pub tool_call_count: usize,
    #[serde(default)]
    pub call_site_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorFrame {
    pub schema_version: String,
    pub sensor_frame_id: String,
    pub candidate_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub reward: f64,
    pub status: String,
    #[serde(default)]
    pub success_status: Option<String>,
    #[serde(default)]
    pub objective_scores: Vec<ObjectiveScore>,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub trace_digest: Option<TraceDigest>,
    #[serde(default)]
    pub actionable_side_info: Option<Value>,
    #[serde(default)]
    pub artifact_refs: Vec<ArtifactRef>,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl SensorFrame {
    pub fn from_rollout_response(
        candidate_id: &str,
        row: &Value,
        evaluation_stage: &str,
        response: &Value,
    ) -> Result<Self> {
        let rollout_response = RolloutResponse::from_value(response.clone())?;
        rollout_response.validate_for_gepa()?;
        let example_id = dataset_row_identity(row)?;
        let seed = row
            .get("seed")
            .or_else(|| response.get("seed"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let split = row
            .get("split")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let reward = rollout_response.outcome_reward()?;
        let status = rollout_response.status.clone();
        let success_status = rollout_response.success_status.clone();
        let rollout_id = rollout_response.rollout_id.clone();
        let sensor_frame_id = sensor_frame_id(candidate_id, evaluation_stage, &example_id, seed);
        let mut metadata = Map::new();
        if let Some(details) = rollout_response
            .reward_info
            .as_ref()
            .and_then(|reward_info| reward_info.extra.get("details"))
        {
            metadata.insert("reward_details".to_string(), details.clone());
        }
        if !rollout_response.summary.is_empty() {
            metadata.insert(
                "summary".to_string(),
                Value::Object(rollout_response.summary.clone()),
            );
        }
        let objective = row
            .get("objective")
            .and_then(Value::as_str)
            .unwrap_or("outcome_reward")
            .to_string();
        let failure = if status == "completed" || status == "succeeded" {
            None
        } else {
            Some(
                FailurePayload::new(
                    OptimizerFailureType::Container,
                    "rollout_not_completed",
                    format!("rollout returned status {status}"),
                    false,
                )
                .with_detail("status", Value::String(status.clone())),
            )
        };
        Ok(Self {
            schema_version: "sensor_frame.v1".to_string(),
            sensor_frame_id,
            candidate_id: candidate_id.to_string(),
            rollout_id,
            example_id,
            seed,
            split,
            evaluation_stage: evaluation_stage.to_string(),
            reward,
            status,
            success_status,
            objective_scores: vec![ObjectiveScore {
                objective,
                value: reward,
                source: "container.reward_info".to_string(),
                rationale: rollout_response
                    .reward_info
                    .as_ref()
                    .and_then(|reward_info| reward_info.extra.get("details"))
                    .and_then(|details| details.get("objective"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                metadata: Map::new(),
            }],
            usage: if rollout_response.usage.is_empty() {
                json!({})
            } else {
                Value::Object(rollout_response.usage.clone())
            },
            trace_digest: rollout_response.trace.as_ref().map(trace_digest),
            actionable_side_info: response.get("actionable_side_info").cloned(),
            artifact_refs: Vec::new(),
            failure,
            metadata,
        })
    }
}

fn sensor_frame_id(
    candidate_id: &str,
    evaluation_stage: &str,
    example_id: &str,
    seed: i64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(candidate_id.as_bytes());
    digest.update(evaluation_stage.as_bytes());
    digest.update(example_id.as_bytes());
    digest.update(seed.to_string().as_bytes());
    let hex = format!("{:x}", digest.finalize());
    format!("sensor_{}", &hex[..12])
}

fn trace_digest(trace: &Value) -> TraceDigest {
    let mut digest = Sha256::new();
    digest.update(stable_json(trace).as_bytes());
    let mut call_site_ids = Vec::new();
    let mut llm_request_count = 0usize;
    let mut tool_call_count = 0usize;
    let event_history = trace
        .get("event_history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for event in &event_history {
        if event.get("llm_request").is_some()
            || event.get("type").and_then(Value::as_str) == Some("llm_request")
        {
            llm_request_count += 1;
        }
        if event.get("tool_call").is_some()
            || event.get("type").and_then(Value::as_str) == Some("tool_call")
        {
            tool_call_count += 1;
        }
        if let Some(call_site_id) = event.get("call_site_id").and_then(Value::as_str) {
            call_site_ids.push(call_site_id.to_string());
        }
    }
    call_site_ids.sort();
    call_site_ids.dedup();
    TraceDigest {
        sha256: format!("{:x}", digest.finalize()),
        event_count: event_history.len(),
        llm_request_count,
        tool_call_count,
        call_site_ids,
    }
}
