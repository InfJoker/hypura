use serde::{Deserialize, Serialize};

// ── Tool types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ── Request types ──

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: GenerateOptions,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: GenerateOptions,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Default, Deserialize)]
pub struct GenerateOptions {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub num_predict: Option<u32>,
    pub seed: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ShowRequest {
    pub model: String,
}

// ── Response types ──

#[derive(Debug, Serialize)]
pub struct GenerateResponseChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten)]
    pub timing: TimingStats,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseChunk {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(flatten)]
    pub timing: TimingStats,
}

#[derive(Debug, Serialize)]
pub struct TagsResponse {
    pub models: Vec<ModelTag>,
}

#[derive(Debug, Serialize)]
pub struct ModelTag {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub details: ModelDetails,
}

#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize)]
pub struct ShowResponse {
    pub details: ModelDetails,
    pub model_info: serde_json::Value,
}

/// Timing statistics shared by generate and chat response types.
#[derive(Debug, Default, Serialize)]
pub struct TimingStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

impl TimingStats {
    pub fn from_result(
        result: &Option<crate::compute::inference::GenerationResult>,
        total_ns: u64,
        load_duration_ns: u64,
    ) -> Self {
        Self {
            total_duration: Some(total_ns),
            load_duration: Some(load_duration_ns),
            prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
            prompt_eval_duration: result
                .as_ref()
                .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
            eval_count: result.as_ref().map(|r| r.tokens_generated),
            eval_duration: result.as_ref().map(|r| {
                if r.tok_per_sec_avg > 0.0 {
                    (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
                } else {
                    0
                }
            }),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Produce an RFC 3339 timestamp with nanosecond precision.
pub fn now_rfc3339() -> String {
    chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

/// Lightweight GGUF info kept in AppState (no tensor data).
#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub file_size: u64,
    pub architecture: String,
    pub parameter_count: u64,
    pub quantization: String,
    pub context_length: u32,
}
