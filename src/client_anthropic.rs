//! Anthropic Messages API streaming client.
//!
//! ARCHITECTURE
//! ============
//! Sends a streaming request to `/v1/messages` with `stream: true` and decodes
//! the SSE event stream into `ContentDelta` values as chunks arrive.
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
use crate::types::{ContentBlock, ContentDelta, Message, Tool};

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

    /// Stream a chat request, invoking `on_delta` for each decoded delta.
    pub async fn stream_chat<F, Fut>(
        &self,
        model: &str,
        max_tokens: u32,
        system: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
        mut on_delta: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(ContentDelta) -> Fut,
        Fut: std::future::Future<Output = Result<(), LlmError>>,
    {
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

        let mut parser = AnthropicStreamParser::default();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::ApiRequest(e.to_string()))?;
            for delta in parser.push_chunk(&chunk)? {
                on_delta(delta).await?;
            }
        }

        Ok(())
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

#[derive(Default)]
struct AnthropicStreamParser {
    buffer: Vec<u8>,
    model: String,
    in_tokens: u64,
    out_tokens: u64,
    tool_states: std::collections::HashMap<usize, ToolCallState>,
}

#[derive(Default)]
struct ToolCallState {
    id: String,
    name: String,
}

impl AnthropicStreamParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<ContentDelta>, LlmError> {
        self.buffer.extend_from_slice(chunk);
        let mut deltas = Vec::new();

        while let Some(block_end) = self.buffer.windows(2).position(|w| w == b"\n\n") {
            let block = self.buffer.drain(..block_end + 2).collect::<Vec<_>>();
            let block = &block[..block.len().saturating_sub(2)];
            if block.is_empty() {
                continue;
            }
            let block = std::str::from_utf8(block).map_err(|e| LlmError::StreamDecode(e.to_string()))?;
            if let Some(delta) = self.parse_block(block)? {
                deltas.push(delta);
            }
        }

        Ok(deltas)
    }

    fn parse_block(&mut self, block: &str) -> Result<Option<ContentDelta>, LlmError> {
        let event_type = extract_sse_field(block, "event");
        let data = extract_sse_field(block, "data");

        if data == "[DONE]" || event_type == "message_stop" {
            return Ok(None);
        }

        match event_type.as_str() {
            "message_start" => {
                let v = serde_json::from_str::<Value>(&data).map_err(|e| LlmError::StreamDecode(e.to_string()))?;
                if let Some(m) = v.pointer("/message/model").and_then(Value::as_str) {
                    self.model = m.to_string();
                }
                if let Some(t) = v.pointer("/message/usage/input_tokens").and_then(Value::as_u64) {
                    self.in_tokens = t;
                }
                Ok(None)
            }
            "content_block_start" => {
                let v = serde_json::from_str::<Value>(&data).map_err(|e| LlmError::StreamDecode(e.to_string()))?;
                let kind = v.pointer("/content_block/type").and_then(Value::as_str).unwrap_or("");
                if kind != "tool_use" {
                    return Ok(None);
                }
                let index = v.get("index").and_then(Value::as_u64).map_or(0, |n| n as usize);
                let id = v.pointer("/content_block/id").and_then(Value::as_str).unwrap_or("").to_string();
                let name = v.pointer("/content_block/name").and_then(Value::as_str).unwrap_or("").to_string();
                let state = self.tool_states.entry(index).or_default();
                state.id = id.clone();
                state.name = name.clone();
                Ok(Some(ContentDelta::ToolUseDelta { index, id, name, input_fragment: String::new() }))
            }
            "content_block_delta" => {
                let v = serde_json::from_str::<Value>(&data).map_err(|e| LlmError::StreamDecode(e.to_string()))?;
                let delta = match v.get("delta") {
                    Some(delta) => delta,
                    None => return Ok(None),
                };
                let delta_type = delta.get("type").and_then(Value::as_str).unwrap_or("");
                match delta_type {
                    "text_delta" => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                        Ok(Some(ContentDelta::TextDelta(text)))
                    }
                    "thinking_delta" => {
                        let thinking = delta.get("thinking").and_then(Value::as_str).unwrap_or("").to_string();
                        Ok(Some(ContentDelta::ThinkingDelta(thinking)))
                    }
                    "input_json_delta" => {
                        let index = v.get("index").and_then(Value::as_u64).map_or(0, |n| n as usize);
                        let state = self.tool_states.entry(index).or_default();
                        let fragment = delta.get("partial_json").and_then(Value::as_str).unwrap_or("").to_string();
                        Ok(Some(ContentDelta::ToolUseDelta {
                            index,
                            id: state.id.clone(),
                            name: state.name.clone(),
                            input_fragment: fragment,
                        }))
                    }
                    _ => Ok(None),
                }
            }
            "message_delta" => {
                let v = serde_json::from_str::<Value>(&data).map_err(|e| LlmError::StreamDecode(e.to_string()))?;
                let stop_reason = v
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .unwrap_or("end_turn")
                    .to_string();
                self.out_tokens = v.pointer("/usage/output_tokens").and_then(Value::as_u64).unwrap_or(0);
                Ok(Some(ContentDelta::Done {
                    stop_reason,
                    model: self.model.clone(),
                    input_tokens: self.in_tokens,
                    output_tokens: self.out_tokens,
                }))
            }
            _ => Ok(None),
        }
    }
}

fn extract_sse_field(block: &str, field: &str) -> String {
    let prefix = format!("{field}:");
    block
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n")
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
    let mut tool_order: Vec<usize> = Vec::new();
    let mut tool_states: std::collections::HashMap<usize, ToolBlockState> = std::collections::HashMap::new();

    for delta in deltas {
        match delta {
            ContentDelta::TextDelta(t) => {
                flush_thinking(&mut thinking_buf, &mut blocks);
                flush_tools(&mut tool_order, &mut tool_states, &mut blocks);
                text_buf.push_str(t);
            }
            ContentDelta::ThinkingDelta(t) => {
                flush_text(&mut text_buf, &mut blocks);
                flush_tools(&mut tool_order, &mut tool_states, &mut blocks);
                thinking_buf.push_str(t);
            }
            ContentDelta::ToolUseDelta { index, id, name, input_fragment } => {
                flush_text(&mut text_buf, &mut blocks);
                flush_thinking(&mut thinking_buf, &mut blocks);
                let state = tool_states.entry(*index).or_insert_with(|| {
                    tool_order.push(*index);
                    ToolBlockState::default()
                });
                if !id.is_empty() {
                    state.id = id.clone();
                }
                if !name.is_empty() {
                    state.name = name.clone();
                }
                state.input_buf.push_str(input_fragment);
            }
            ContentDelta::Done { stop_reason: sr, .. } => {
                stop_reason = Some(sr.clone());
            }
        }
    }

    flush_text(&mut text_buf, &mut blocks);
    flush_thinking(&mut thinking_buf, &mut blocks);
    flush_tools(&mut tool_order, &mut tool_states, &mut blocks);

    (blocks, stop_reason)
}

#[derive(Default)]
struct ToolBlockState {
    id: String,
    name: String,
    input_buf: String,
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

fn flush_tools(
    tool_order: &mut Vec<usize>,
    tool_states: &mut std::collections::HashMap<usize, ToolBlockState>,
    blocks: &mut Vec<ContentBlock>,
) {
    for index in std::mem::take(tool_order) {
        let Some(state) = tool_states.remove(&index) else { continue };
        if state.id.is_empty() {
            continue;
        }
        let input = serde_json::from_str::<Value>(&state.input_buf)
            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
        blocks.push(ContentBlock::ToolUse { id: state.id, name: state.name, input });
    }
}

#[cfg(test)]
mod tests {
    use super::{AnthropicStreamParser, reconstruct_content_blocks};
    use crate::types::{ContentBlock, ContentDelta};

    #[test]
    fn parser_keeps_tool_fragments_separate_by_index() {
        let mut parser = AnthropicStreamParser::default();
        let chunk = concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"lookup\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_2\",\"name\":\"search\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"b\\\":\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"1}\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"2}\"}}\n\n"
        );

        let deltas = parser.push_chunk(chunk.as_bytes()).unwrap();
        let (blocks, _) = reconstruct_content_blocks(&deltas);
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { id, .. } if id == "tool_1"));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { id, .. } if id == "tool_2"));
    }

    #[test]
    fn reconstruct_content_blocks_handles_multiple_tool_indices() {
        let deltas = vec![
            ContentDelta::ToolUseDelta {
                index: 0,
                id: "tool_1".to_string(),
                name: "lookup".to_string(),
                input_fragment: "{\"a\":".to_string(),
            },
            ContentDelta::ToolUseDelta {
                index: 1,
                id: "tool_2".to_string(),
                name: "search".to_string(),
                input_fragment: "{\"b\":".to_string(),
            },
            ContentDelta::ToolUseDelta {
                index: 0,
                id: "tool_1".to_string(),
                name: "lookup".to_string(),
                input_fragment: "1}".to_string(),
            },
            ContentDelta::ToolUseDelta {
                index: 1,
                id: "tool_2".to_string(),
                name: "search".to_string(),
                input_fragment: "2}".to_string(),
            },
        ];

        let (blocks, _) = reconstruct_content_blocks(&deltas);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { id, .. } if id == "tool_1"));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { id, .. } if id == "tool_2"));
    }
}
