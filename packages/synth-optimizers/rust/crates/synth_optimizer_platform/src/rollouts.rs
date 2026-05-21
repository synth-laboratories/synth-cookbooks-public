use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::failures::FailurePayload;
use crate::sensors::SensorFrame;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutRecord {
    pub schema_version: String,
    pub rollout_record_id: String,
    #[serde(default)]
    pub rollout_id: Option<String>,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub status: String,
    pub reward: f64,
    #[serde(default)]
    pub trace_sha256: Option<String>,
    pub event_count: u64,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub failure: Option<FailurePayload>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutEventRecord {
    pub schema_version: String,
    pub rollout_event_id: String,
    pub rollout_record_id: String,
    pub candidate_id: String,
    pub sensor_frame_id: String,
    pub sequence_number: u64,
    pub event_type: String,
    pub summary: String,
    pub event: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorRolloutRecords {
    pub rollout: RolloutRecord,
    pub events: Vec<RolloutEventRecord>,
}

impl SensorRolloutRecords {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let rollout = RolloutRecord::from_sensor_frame(frame);
        let events = RolloutEventRecord::from_sensor_frame(frame, &rollout);
        Self { rollout, events }
    }
}

impl RolloutRecord {
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let trace_digest = frame.trace_digest.as_ref();
        Self {
            schema_version: "rollout_record.v1".to_string(),
            rollout_record_id: stable_id(
                "rolloutrec",
                &[
                    &frame.candidate_id,
                    &frame.sensor_frame_id,
                    &frame.evaluation_stage,
                ],
            ),
            rollout_id: frame.rollout_id.clone(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            example_id: frame.example_id.clone(),
            seed: frame.seed,
            split: frame.split.clone(),
            evaluation_stage: frame.evaluation_stage.clone(),
            status: frame.status.clone(),
            reward: frame.reward,
            trace_sha256: trace_digest.map(|digest| digest.sha256.clone()),
            event_count: trace_digest
                .map(|digest| digest.event_count as u64)
                .unwrap_or(0),
            usage: frame.usage.clone(),
            failure: frame.failure.clone(),
            metadata: frame.metadata.clone(),
        }
    }
}

impl RolloutEventRecord {
    pub fn from_sensor_frame(frame: &SensorFrame, rollout: &RolloutRecord) -> Vec<Self> {
        let mut records = vec![Self {
            schema_version: "rollout_event_record.v1".to_string(),
            rollout_event_id: stable_id(
                "rolloutev",
                &[&rollout.rollout_record_id, "1", "rollout_observed"],
            ),
            rollout_record_id: rollout.rollout_record_id.clone(),
            candidate_id: frame.candidate_id.clone(),
            sensor_frame_id: frame.sensor_frame_id.clone(),
            sequence_number: 1,
            event_type: "rollout_observed".to_string(),
            summary: format!(
                "rollout {} for {} completed with status {} and reward {:.6}",
                frame.evaluation_stage, frame.example_id, frame.status, frame.reward
            ),
            event: json!({
                "candidate_id": &frame.candidate_id,
                "sensor_frame_id": &frame.sensor_frame_id,
                "rollout_id": &frame.rollout_id,
                "example_id": &frame.example_id,
                "seed": frame.seed,
                "split": &frame.split,
                "evaluation_stage": &frame.evaluation_stage,
                "status": &frame.status,
                "success_status": &frame.success_status,
                "reward": frame.reward,
                "failure": &frame.failure,
            }),
        }];
        if let Some(trace_digest) = &frame.trace_digest {
            records.push(Self {
                schema_version: "rollout_event_record.v1".to_string(),
                rollout_event_id: stable_id(
                    "rolloutev",
                    &[&rollout.rollout_record_id, "2", "trace_digest_observed"],
                ),
                rollout_record_id: rollout.rollout_record_id.clone(),
                candidate_id: frame.candidate_id.clone(),
                sensor_frame_id: frame.sensor_frame_id.clone(),
                sequence_number: 2,
                event_type: "trace_digest_observed".to_string(),
                summary: format!(
                    "trace digest has {} events, {} llm requests, and {} tool calls",
                    trace_digest.event_count,
                    trace_digest.llm_request_count,
                    trace_digest.tool_call_count
                ),
                event: json!({
                    "trace_sha256": &trace_digest.sha256,
                    "event_count": trace_digest.event_count,
                    "llm_request_count": trace_digest.llm_request_count,
                    "tool_call_count": trace_digest.tool_call_count,
                    "call_site_ids": &trace_digest.call_site_ids,
                }),
            });
        }
        records
    }
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}
