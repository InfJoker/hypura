use std::time::Instant;

use axum::body::Body;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::compute::inference::{GeneratedToken, GenerationResult};
use crate::server::ollama_types::*;

/// Convert a token channel into an NDJSON streaming body for `/api/generate`.
pub fn ndjson_generate_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        // Stream token chunks
        while let Some(token) = token_rx.recv().await {
            let chunk = GenerateResponseChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                response: token.text,
                done: false,
                done_reason: None,
                timing: TimingStats::default(),
            };
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }

        // Final chunk with timing
        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        let final_chunk = GenerateResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: String::new(),
            done: true,
            done_reason: Some("stop".into()),
            timing: TimingStats::from_result(&result, total_ns, load_duration_ns),
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}

/// Convert a token channel into an NDJSON streaming body for `/api/chat`.
///
/// Tool call parsing happens in the final chunk after all tokens are collected,
/// matching Ollama's streaming behavior.
pub fn ndjson_chat_stream(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
    tools: Vec<crate::server::ollama_types::Tool>,
) -> Body {
    let (tx, rx) = mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut full_response = String::new();

        while let Some(token) = token_rx.recv().await {
            full_response.push_str(&token.text);
            let chunk = ChatResponseChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(token.text),
                    tool_calls: None,
                },
                done: false,
                done_reason: None,
                timing: TimingStats::default(),
            };
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }

        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();

        // Parse tool calls in the final chunk if tools were provided
        let (content, tool_calls, done_reason) = if !tools.is_empty() {
            use crate::server::tool_parse::{parse_tool_calls, ToolParseResult};
            match parse_tool_calls(&full_response, &tools) {
                ToolParseResult::ToolCalls { content, calls } => {
                    let c = if content.is_empty() { None } else { Some(content) };
                    (c, Some(calls), "tool_calls")
                }
                ToolParseResult::Text(t) => (Some(t), None, "stop"),
            }
        } else {
            (Some(String::new()), None, "stop")
        };

        let final_chunk = ChatResponseChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: ChatMessage {
                role: "assistant".into(),
                content,
                tool_calls,
            },
            done: true,
            done_reason: Some(done_reason.into()),
            timing: TimingStats::from_result(&result, total_ns, load_duration_ns),
        };
        let mut line = serde_json::to_string(&final_chunk).unwrap_or_default();
        line.push('\n');
        let _ = tx.send(Ok(line)).await;
    });

    Body::from_stream(ReceiverStream::new(rx))
}
