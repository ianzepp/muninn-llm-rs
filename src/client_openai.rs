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

use futures_util::StreamExt;
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

/// HTTP client for the OpenAI-compatible chat completions API.
///
/// One instance is created per config profile at `LlmSyscall` construction time
/// and reused across all requests for that profile. `base_url` defaults to the
/// official OpenAI endpoint but can be overridden to point at any compatible API
/// (e.g., Azure OpenAI, local Ollama, or other OpenAI-compatible services).
pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAiClient {
    /// Construct an `OpenAiClient` with the given API key and optional base URL.
    ///
    /// `base_url` defaults to `https://api.openai.com` when `None`. Trailing
    /// slashes are stripped so path construction is consistent.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::HttpClientBuild`] if the `reqwest` client cannot be
    /// constructed.
    pub fn new(api_key: String, base_url: Option<&str>) -> Result<Self, LlmError> {
        let base_url = base_url
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_string();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| LlmError::HttpClientBuild(e.to_string()))?;
        Ok(Self {
            http,
            api_key,
            base_url,
        })
    }

    /// Stream a chat request. Returns collected `ContentDelta` values.
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
        let cc_messages = build_cc_messages(system, messages);
        let tool_defs: Option<Vec<CcToolDef<'_>>> =
            tools.map(|t| t.iter().map(CcToolDef::from_tool).collect());

        let body = CcRequest {
            model,
            max_tokens,
            messages: &cc_messages,
            tools: tool_defs.as_deref(),
            stream: true,
            stream_options: Some(CcStreamOptions {
                include_usage: true,
            }),
        };

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
            let body_text = response
                .text()
                .await
                .map_err(|e| LlmError::ApiRequest(e.to_string()))?;
            return Err(LlmError::ApiResponse {
                status,
                body: body_text,
            });
        }

        let mut parser = OpenAiStreamParser::default();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::ApiRequest(e.to_string()))?;
            for delta in parser.push_chunk(&chunk)? {
                on_delta(delta).await?;
            }
        }

        if let Some(done) = parser.finish()? {
            on_delta(done).await?;
        }

        Ok(())
    }
}

// =============================================================================
// WIRE TYPES
// =============================================================================

/// OpenAI `/v1/chat/completions` request body.
#[derive(Serialize)]
struct CcRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [CcMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [CcToolDef<'a>]>,
    stream: bool,
    /// WHY include_usage: OpenAI only emits token counts in a trailing
    /// chunk when this option is set. Without it we cannot populate
    /// `ContentDelta::Done.input_tokens` / `output_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<CcStreamOptions>,
}

#[derive(Serialize)]
struct CcStreamOptions {
    include_usage: bool,
}

/// A single message in the OpenAI chat completions wire format.
///
/// All fields are optional at the struct level because different message
/// kinds populate different subsets: text messages set `content`, assistant
/// messages with tool calls set `tool_calls`, and tool result messages set
/// both `content` and `tool_call_id`.
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

/// A single tool call reference in an assistant message.
#[derive(Serialize)]
struct CcToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: CcFunction,
}

/// The function name and JSON-serialized arguments for a tool call.
#[derive(Serialize)]
struct CcFunction {
    name: String,
    /// WHY String not Value: OpenAI requires arguments as a JSON *string*,
    /// not an inline object. We serialize the input Value to a string here.
    arguments: String,
}

/// OpenAI tool definition wire shape (wraps a function definition).
#[derive(Serialize)]
struct CcToolDef<'a> {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: CcFunctionDef<'a>,
}

/// OpenAI function definition — name, description, and JSON Schema parameters.
///
/// Uses borrowed references to avoid cloning the `input_schema` JSON value.
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

/// Convert a system prompt and provider-neutral messages into the OpenAI wire format.
///
/// WHY system-as-message: Unlike Anthropic (which has a top-level `system` field),
/// OpenAI embeds the system prompt as the first message with `role: "system"`.
/// This function handles that injection and also translates `Content::Blocks`
/// (which may contain tool calls and tool results) into the OpenAI multi-message
/// format where each tool result is a separate message with `role: "tool"`.
fn build_cc_messages(system: &str, messages: &[Message]) -> Vec<CcMessage> {
    let mut out = Vec::new();
    if !system.trim().is_empty() {
        out.push(CcMessage {
            role: "system".to_string(),
            content: Some(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    for msg in messages {
        match &msg.content {
            Content::Text(text) => {
                out.push(CcMessage {
                    role: msg.role.clone(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
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
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
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
                        content: if text_buf.is_empty() {
                            None
                        } else {
                            Some(text_buf)
                        },
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
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

/// Incremental NDJSON/SSE parser for the OpenAI streaming response.
///
/// OpenAI sends `data: <json>\n` lines. The parser accumulates raw bytes in
/// `buffer`, extracts complete lines, and translates each JSON object to zero
/// or more [`ContentDelta`] values.
///
/// Token counts are emitted in a trailing chunk (when `stream_options.include_usage`
/// is set) and stored until `take_done` assembles the final `ContentDelta::Done`.
/// The stop reason arrives in the `finish_reason` field of the choices array and
/// is stored as `pending_stop_reason` until the usage chunk confirms the stream end.
#[derive(Default)]
struct OpenAiStreamParser {
    buffer: Vec<u8>,
    model: String,
    in_tokens: u64,
    out_tokens: u64,
    /// Tracks id and name for each in-progress tool call by array index.
    tool_states: std::collections::HashMap<usize, ToolCallState>,
    /// Stop reason received from a choice's `finish_reason` field, held until
    /// the usage chunk arrives so both can be combined into a single `Done` delta.
    pending_stop_reason: Option<String>,
}

/// Accumulated state for a single in-progress tool call delta.
#[derive(Default)]
struct ToolCallState {
    id: String,
    name: String,
}

impl OpenAiStreamParser {
    fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<ContentDelta>, LlmError> {
        self.buffer.extend_from_slice(chunk);
        let mut deltas = Vec::new();

        while let Some(line_end) = self.buffer.iter().position(|b| *b == b'\n') {
            let mut line = self.buffer.drain(..=line_end).collect::<Vec<_>>();
            if matches!(line.last(), Some(b'\n')) {
                line.pop();
            }
            if matches!(line.last(), Some(b'\r')) {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            let line = std::str::from_utf8(&line)
                .map_err(|e| LlmError::StreamDecode(e.to_string()))?
                .trim();
            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            deltas.extend(self.parse_event(data)?);
        }

        Ok(deltas)
    }

    fn finish(&mut self) -> Result<Option<ContentDelta>, LlmError> {
        if self.buffer.is_empty() {
            return Ok(self.take_done());
        }
        let trailing = std::str::from_utf8(&self.buffer)
            .map_err(|e| LlmError::StreamDecode(e.to_string()))?
            .trim()
            .to_string();
        self.buffer.clear();
        if trailing.is_empty() || trailing == "data: [DONE]" {
            return Ok(self.take_done());
        }
        Ok(None)
    }

    fn parse_event(&mut self, data: &str) -> Result<Vec<ContentDelta>, LlmError> {
        let v = serde_json::from_str::<Value>(data)
            .map_err(|e| LlmError::StreamDecode(e.to_string()))?;
        let mut deltas = Vec::new();

        if let Some(m) = v.get("model").and_then(Value::as_str) {
            self.model = m.to_string();
        }
        if let Some(t) = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64) {
            self.in_tokens = t;
        }
        if let Some(t) = v
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            self.out_tokens = t;
        }

        if let Some(choice) = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
        {
            if let Some(delta) = choice.get("delta") {
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        deltas.push(ContentDelta::TextDelta(text.to_string()));
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for (fallback_index, call) in tool_calls.iter().enumerate() {
                        let index = call
                            .get("index")
                            .and_then(Value::as_u64)
                            .and_then(|n| usize::try_from(n).ok())
                            .unwrap_or(fallback_index);
                        let state = self.tool_states.entry(index).or_default();

                        if let Some(id) = call.get("id").and_then(Value::as_str) {
                            state.id = id.to_string();
                        }
                        if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                            state.name = name.to_string();
                        }

                        let fragment = call
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if !state.id.is_empty() || !state.name.is_empty() || !fragment.is_empty() {
                            deltas.push(ContentDelta::ToolUseDelta {
                                index,
                                id: state.id.clone(),
                                name: state.name.clone(),
                                input_fragment: fragment.to_string(),
                            });
                        }
                    }
                }
            }

            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.pending_stop_reason = Some(match reason {
                    "tool_calls" => "tool_use".to_string(),
                    "length" => "max_tokens".to_string(),
                    _ => "end_turn".to_string(),
                });
            }
        }

        if v.get("choices")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            if let Some(done) = self.take_done() {
                deltas.push(done);
            }
        }

        Ok(deltas)
    }

    fn take_done(&mut self) -> Option<ContentDelta> {
        self.pending_stop_reason
            .take()
            .map(|stop_reason| ContentDelta::Done {
                stop_reason,
                model: self.model.clone(),
                input_tokens: self.in_tokens,
                output_tokens: self.out_tokens,
            })
    }
}

#[cfg(test)]
#[path = "client_openai_test.rs"]
mod tests;
