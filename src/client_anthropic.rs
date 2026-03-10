//! Anthropic Messages API streaming client.
//!
//! ARCHITECTURE
//! ============
//! Sends a streaming request to `/v1/messages` with `stream: true` and decodes
//! the SSE event stream into `ContentDelta` values.
//!
//! SSE EVENT MAPPING
//! =================
//! - `content_block_start` (tool_use) → `ToolUseDelta { id, name, input: "" }`
//! - `content_block_delta` (text_delta) → `TextDelta`
//! - `content_block_delta` (thinking_delta) → `ThinkingDelta`
//! - `content_block_delta` (input_json_delta) → `ToolUseDelta { fragment }`
//! - `message_delta` (stop_reason, usage) → `Done`
//! - `message_stop` → stream end

use std::time::Duration;

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::{Content, ContentBlock, ContentDelta, Message, Tool};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const BETA_HEADER: &str = "interleaved-thinking-2025-05-14";
const REQUEST_TIMEOUT_SECS: u64 = 300;
const CONNECT_TIMEOUT_SECS: u64 = 10;

// =============================================================================
// CLIENT
// =============================================================================

pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| LlmError::HttpClientBuild(e.to_string()))?;
        Ok(Self { http, api_key })
    }

    /// Stream a chat request. Returns collected `ContentDelta` values.
    ///
    /// WHY collect-then-return instead of `impl Stream`: the stream returned by
    /// `reqwest` involves complex lifetime and Pin constraints. Collecting here
    /// keeps the call-site simple while still giving the room loop each delta
    /// in order. True incremental streaming to frames is done in `llm/chat.rs`
    /// where deltas are forwarded as they arrive.
    pub async fn stream_chat(
        &self,
        model: &str,
        max_tokens: u32,
        system: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
    ) -> Result<Vec<ContentDelta>, LlmError> {
        let tool_defs: Option<Vec<AnthropicTool<'_>>> =
            tools.map(|t| t.iter().map(AnthropicTool::from_tool).collect());

        let body = ApiRequest {
            model,
            max_tokens,
            system,
            messages,
            tools: tool_defs.as_deref(),
            stream: true,
        };

        let response = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("anthropic-beta", BETA_HEADER)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::ApiRequest(e.to_string()))?;

        let status = response.status().as_u16();
        if status != 200 {
            let body_text = response.text().await.map_err(|e| LlmError::ApiRequest(e.to_string()))?;
            return Err(LlmError::ApiResponse { status, body: body_text });
        }

        let text = response.text().await.map_err(|e| LlmError::ApiRequest(e.to_string()))?;
        decode_sse_text(&text)
    }
}

// =============================================================================
// WIRE TYPES
// =============================================================================

#[derive(Serialize)]
struct ApiRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [AnthropicTool<'a>]>,
    stream: bool,
}

#[derive(Serialize)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a Value,
}

impl<'a> AnthropicTool<'a> {
    fn from_tool(tool: &'a Tool) -> Self {
        Self { name: &tool.name, description: &tool.description, input_schema: &tool.input_schema }
    }
}

// =============================================================================
// SSE DECODING
// =============================================================================

fn decode_sse_text(text: &str) -> Result<Vec<ContentDelta>, LlmError> {
    let mut deltas = Vec::new();
    let mut model = String::new();
    let mut tool_id = String::new();
    let mut in_tokens: u64 = 0;
    let mut out_tokens: u64 = 0;

    for block in text.split("\n\n") {
        let event_type = extract_sse_field(block, "event");
        let data = extract_sse_field(block, "data");

        if data == "[DONE]" || event_type == "message_stop" {
            break;
        }

        if let Some(delta) = parse_sse_event(
            &event_type,
            &data,
            &mut model,
            &mut tool_id,
            &mut in_tokens,
            &mut out_tokens,
        ) {
            deltas.push(delta?);
        }
    }

    Ok(deltas)
}

fn extract_sse_field(block: &str, field: &str) -> String {
    let prefix = format!("{field}: ");
    block
        .lines()
        .find(|line| line.starts_with(&prefix))
        .map(|line| line[prefix.len()..].to_string())
        .unwrap_or_default()
}

fn parse_sse_event(
    event_type: &str,
    data: &str,
    model: &mut String,
    tool_id: &mut String,
    in_tokens: &mut u64,
    out_tokens: &mut u64,
) -> Option<Result<ContentDelta, LlmError>> {
    match event_type {
        "message_start" => {
            let v = serde_json::from_str::<Value>(data).ok()?;
            if let Some(m) = v.pointer("/message/model").and_then(Value::as_str) {
                *model = m.to_string();
            }
            if let Some(t) = v.pointer("/message/usage/input_tokens").and_then(Value::as_u64) {
                *in_tokens = t;
            }
            None
        }
        "content_block_start" => {
            let v = serde_json::from_str::<Value>(data).ok()?;
            let kind = v.pointer("/content_block/type").and_then(Value::as_str).unwrap_or("");
            if kind == "tool_use" {
                let id = v.pointer("/content_block/id").and_then(Value::as_str).unwrap_or("").to_string();
                let name = v.pointer("/content_block/name").and_then(Value::as_str).unwrap_or("").to_string();
                *tool_id = id.clone();
                Some(Ok(ContentDelta::ToolUseDelta { id, name, input_fragment: String::new() }))
            } else {
                None
            }
        }
        "content_block_delta" => {
            let v = serde_json::from_str::<Value>(data).ok()?;
            let delta = v.get("delta")?;
            let delta_type = delta.get("type").and_then(Value::as_str).unwrap_or("");
            match delta_type {
                "text_delta" => {
                    let text = delta.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                    Some(Ok(ContentDelta::TextDelta(text)))
                }
                "thinking_delta" => {
                    let thinking = delta.get("thinking").and_then(Value::as_str).unwrap_or("").to_string();
                    Some(Ok(ContentDelta::ThinkingDelta(thinking)))
                }
                "input_json_delta" => {
                    let fragment = delta.get("partial_json").and_then(Value::as_str).unwrap_or("").to_string();
                    Some(Ok(ContentDelta::ToolUseDelta {
                        id: tool_id.clone(),
                        name: String::new(),
                        input_fragment: fragment,
                    }))
                }
                _ => None,
            }
        }
        "message_delta" => {
            let v = serde_json::from_str::<Value>(data).ok()?;
            let stop_reason = v
                .pointer("/delta/stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_string();
            *out_tokens = v.pointer("/usage/output_tokens").and_then(Value::as_u64).unwrap_or(0);
            Some(Ok(ContentDelta::Done {
                stop_reason,
                model: model.clone(),
                input_tokens: *in_tokens,
                output_tokens: *out_tokens,
            }))
        }
        _ => None,
    }
}

// =============================================================================
// BLOCK RECONSTRUCTION
// =============================================================================

/// Reconstruct `ContentBlock`s from an ordered list of deltas.
///
/// Text and thinking deltas are concatenated. Tool use deltas assemble input
/// JSON from fragments. Returns (blocks, stop_reason).
pub fn reconstruct_content_blocks(deltas: &[ContentDelta]) -> (Vec<ContentBlock>, Option<String>) {
    let mut blocks: Vec<ContentBlock> = Vec::new();
    let mut stop_reason: Option<String> = None;

    let mut text_buf = String::new();
    let mut thinking_buf = String::new();
    let mut tool_id = String::new();
    let mut tool_name = String::new();
    let mut tool_input_buf = String::new();

    for delta in deltas {
        match delta {
            ContentDelta::TextDelta(t) => {
                flush_thinking(&mut thinking_buf, &mut blocks);
                flush_tool(&mut tool_id, &mut tool_name, &mut tool_input_buf, &mut blocks);
                text_buf.push_str(t);
            }
            ContentDelta::ThinkingDelta(t) => {
                flush_text(&mut text_buf, &mut blocks);
                flush_tool(&mut tool_id, &mut tool_name, &mut tool_input_buf, &mut blocks);
                thinking_buf.push_str(t);
            }
            ContentDelta::ToolUseDelta { id, name, input_fragment } => {
                flush_text(&mut text_buf, &mut blocks);
                flush_thinking(&mut thinking_buf, &mut blocks);
                if !id.is_empty() {
                    flush_tool(&mut tool_id, &mut tool_name, &mut tool_input_buf, &mut blocks);
                    tool_id = id.clone();
                    tool_name = name.clone();
                }
                tool_input_buf.push_str(input_fragment);
            }
            ContentDelta::Done { stop_reason: sr, .. } => {
                stop_reason = Some(sr.clone());
            }
        }
    }

    flush_text(&mut text_buf, &mut blocks);
    flush_thinking(&mut thinking_buf, &mut blocks);
    flush_tool(&mut tool_id, &mut tool_name, &mut tool_input_buf, &mut blocks);

    (blocks, stop_reason)
}

fn flush_text(buf: &mut String, blocks: &mut Vec<ContentBlock>) {
    if !buf.is_empty() {
        blocks.push(ContentBlock::Text { text: std::mem::take(buf) });
    }
}

fn flush_thinking(buf: &mut String, blocks: &mut Vec<ContentBlock>) {
    if !buf.is_empty() {
        blocks.push(ContentBlock::Thinking { thinking: std::mem::take(buf) });
    }
}

fn flush_tool(id: &mut String, name: &mut String, input_buf: &mut String, blocks: &mut Vec<ContentBlock>) {
    if !id.is_empty() {
        let input = serde_json::from_str::<Value>(input_buf)
            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
        blocks.push(ContentBlock::ToolUse { id: std::mem::take(id), name: std::mem::take(name), input });
        input_buf.clear();
    }
}
