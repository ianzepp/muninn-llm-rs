//! OpenAI-compatible chat completions streaming client.
//!
//! ARCHITECTURE
//! ============
//! Sends a streaming request to `/v1/chat/completions` and decodes the
//! NDJSON `data:` lines into `ContentDelta` values.
//!
//! DESIGN
//! ======
//! - Collects the full response text then parses it, same as `client_anthropic`.
//!   This avoids complex async stream lifetime issues while still supporting
//!   incremental forwarding at the frame layer (handled in `llm/chat.rs`).
//! - `system` is injected as the first message with role "system".
//! - Tool call arguments are JSON strings per OpenAI spec; we parse them back.

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::error::LlmError;
use crate::types::{Content, ContentBlock, ContentDelta, Message, Tool};

const REQUEST_TIMEOUT_SECS: u64 = 300;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

// =============================================================================
// CLIENT
// =============================================================================

pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(api_key: String, base_url: Option<&str>) -> Result<Self, LlmError> {
        let base_url = base_url.unwrap_or(DEFAULT_BASE_URL).trim_end_matches('/').to_string();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| LlmError::HttpClientBuild(e.to_string()))?;
        Ok(Self { http, api_key, base_url })
    }

    /// Stream a chat request. Returns collected `ContentDelta` values.
    pub async fn stream_chat(
        &self,
        model: &str,
        max_tokens: u32,
        system: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
    ) -> Result<Vec<ContentDelta>, LlmError> {
        let cc_messages = build_cc_messages(system, messages);
        let tool_defs: Option<Vec<CcToolDef<'_>>> =
            tools.map(|t| t.iter().map(CcToolDef::from_tool).collect());

        let body = CcRequest { model, max_tokens, messages: &cc_messages, tools: tool_defs.as_deref(), stream: true };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
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
        decode_ndjson_text(&text)
    }
}

// =============================================================================
// WIRE TYPES
// =============================================================================

#[derive(Serialize)]
struct CcRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [CcMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [CcToolDef<'a>]>,
    stream: bool,
}

#[derive(Serialize)]
struct CcMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CcToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct CcToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: CcFunction,
}

#[derive(Serialize)]
struct CcFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct CcToolDef<'a> {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: CcFunctionDef<'a>,
}

#[derive(Serialize)]
struct CcFunctionDef<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
}

impl<'a> CcToolDef<'a> {
    fn from_tool(tool: &'a Tool) -> Self {
        Self {
            tool_type: "function",
            function: CcFunctionDef {
                name: &tool.name,
                description: &tool.description,
                parameters: &tool.input_schema,
            },
        }
    }
}

// =============================================================================
// MESSAGE BUILDING
// =============================================================================

fn build_cc_messages(system: &str, messages: &[Message]) -> Vec<CcMessage> {
    let mut out = Vec::new();
    if !system.trim().is_empty() {
        out.push(CcMessage { role: "system".to_string(), content: Some(system.to_string()), tool_calls: None, tool_call_id: None });
    }
    for msg in messages {
        match &msg.content {
            Content::Text(text) => {
                out.push(CcMessage { role: msg.role.clone(), content: Some(text.clone()), tool_calls: None, tool_call_id: None });
            }
            Content::Blocks(blocks) => {
                let mut text_buf = String::new();
                let mut tool_calls = Vec::new();
                let mut tool_results = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => text_buf.push_str(text),
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(CcToolCall {
                                id: id.clone(),
                                call_type: "function",
                                function: CcFunction {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        ContentBlock::ToolResult { tool_use_id, content, .. } => {
                            tool_results.push(CcMessage {
                                role: "tool".to_string(),
                                content: Some(content.clone()),
                                tool_calls: None,
                                tool_call_id: Some(tool_use_id.clone()),
                            });
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
                    }
                }
                if !text_buf.is_empty() || !tool_calls.is_empty() {
                    out.push(CcMessage {
                        role: msg.role.clone(),
                        content: if text_buf.is_empty() { None } else { Some(text_buf) },
                        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                        tool_call_id: None,
                    });
                }
                out.extend(tool_results);
            }
        }
    }
    out
}

// =============================================================================
// NDJSON DECODING
// =============================================================================

fn decode_ndjson_text(text: &str) -> Result<Vec<ContentDelta>, LlmError> {
    let mut deltas = Vec::new();
    let mut model = String::new();
    let mut in_tokens: u64 = 0;
    let mut out_tokens: u64 = 0;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let data = if let Some(rest) = line.strip_prefix("data: ") {
            rest
        } else {
            continue
        };

        if data == "[DONE]" {
            break;
        }

        if let Some(delta) = parse_cc_event(data, &mut model, &mut in_tokens, &mut out_tokens) {
            deltas.push(delta?);
        }
    }

    Ok(deltas)
}

fn parse_cc_event(
    data: &str,
    model: &mut String,
    in_tokens: &mut u64,
    out_tokens: &mut u64,
) -> Option<Result<ContentDelta, LlmError>> {
    let v = serde_json::from_str::<Value>(data).ok()?;

    if let Some(m) = v.get("model").and_then(Value::as_str) {
        *model = m.to_string();
    }
    if let Some(t) = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64) {
        *in_tokens = t;
    }
    if let Some(t) = v.pointer("/usage/completion_tokens").and_then(Value::as_u64) {
        *out_tokens = t;
    }

    let choice = v.get("choices").and_then(Value::as_array).and_then(|a| a.first())?;
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let delta = choice.get("delta")?;

    if let Some(text) = delta.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            return Some(Ok(ContentDelta::TextDelta(text.to_string())));
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("").to_string();
            let name = call.pointer("/function/name").and_then(Value::as_str).unwrap_or("").to_string();
            let fragment = call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("").to_string();
            return Some(Ok(ContentDelta::ToolUseDelta { id, name, input_fragment: fragment }));
        }
    }

    if let Some(reason) = finish_reason {
        let stop_reason = match reason {
            "tool_calls" => "tool_use".to_string(),
            "length" => "max_tokens".to_string(),
            _ => "end_turn".to_string(),
        };
        return Some(Ok(ContentDelta::Done {
            stop_reason,
            model: model.clone(),
            input_tokens: *in_tokens,
            output_tokens: *out_tokens,
        }));
    }

    None
}
