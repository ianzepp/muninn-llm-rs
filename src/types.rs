//! Provider-neutral content types, streaming deltas, and frame conversion helpers.
//!
//! ARCHITECTURE
//! ============
//! All LLM provider clients (`AnthropicClient`, `OpenAiClient`) translate their
//! wire formats into the types defined here. The room loop and syscall handlers
//! work exclusively with these types and never import provider-specific structs.
//!
//! DESIGN
//! ======
//! - `ContentBlock` represents a fully assembled content item (text, tool_use, etc).
//! - `ContentDelta` is a streaming fragment emitted by provider clients during
//!   response streaming. The room loop assembles these back into `ContentBlock`s.
//! - `Content` is the untagged union of string vs block-array used in messages,
//!   matching both Anthropic and OpenAI wire shapes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::LlmError;
use crate::room::state::HistoryEntry;

// =============================================================================
// CONTENT BLOCKS
// =============================================================================

/// A fully assembled content item in a message or API response.
///
/// WHY tagged enum with `#[serde(other)]`: The Anthropic API uses
/// `{"type": "text", ...}` format. `Unknown` catches forward-compatible new
/// block types without failing deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    #[serde(rename = "thinking")]
    Thinking { thinking: String },

    #[serde(other)]
    Unknown,
}

/// A streaming content delta emitted by provider clients.
///
/// Provider SSE/NDJSON streams are decoded into these fragments. The room loop
/// accumulates them into complete `ContentBlock`s and forwards `TextDelta`
/// fragments upstream as `Item` frames immediately for low-latency display.
#[derive(Debug, Clone)]
pub enum ContentDelta {
    TextDelta(String),
    ThinkingDelta(String),
    /// A fragment of a `tool_use` block. `input_fragment` is partial JSON.
    ToolUseDelta {
        index: usize,
        id: String,
        name: String,
        input_fragment: String,
    },
    Done {
        stop_reason: String,
        model: String,
        input_tokens: u64,
        output_tokens: u64,
    },
}

/// Message content — either plain text or structured blocks.
///
/// WHY untagged: Anthropic accepts both `"content": "hello"` (string) and
/// `"content": [{"type": "tool_result", ...}]` (array). Serde tries
/// `Text(String)` first, then `Blocks(Vec)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// =============================================================================
// TOOL DEFINITION
// =============================================================================

/// A tool definition passed to the LLM provider API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// =============================================================================
// MESSAGE
// =============================================================================

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

// =============================================================================
// CHAT REQUEST / RESPONSE
// =============================================================================

/// Request payload for `llm:chat`.
///
/// WHY history + context: Room sends flat history entries (the room's
/// conversation log). The LLM subsystem converts them to provider messages.
/// `context` carries in-progress tool loop turns (assistant responses with
/// `tool_use` blocks and tool results) that Room accumulates between rounds.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub config: String,
    pub history: Vec<HistoryEntry>,
    #[serde(default)]
    pub context: Vec<Message>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

/// Final assembled response from `llm:chat`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// =============================================================================
// FRAME CONVERSION
// =============================================================================

pub type Data = HashMap<String, Value>;

/// Deserialize a typed request from `Frame.data`.
pub fn from_data<T: serde::de::DeserializeOwned>(data: &Data) -> Result<T, LlmError> {
    let map: serde_json::Map<String, Value> =
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    serde_json::from_value(Value::Object(map)).map_err(|e| LlmError::Deserialize(e.to_string()))
}

/// Serialize a typed value into `Frame.data`.
pub fn to_data<T: Serialize>(value: &T) -> Result<Data, LlmError> {
    let v = serde_json::to_value(value).map_err(|e| LlmError::Serialize(e.to_string()))?;
    let Value::Object(map) = v else {
        return Err(LlmError::Serialize("expected object".into()));
    };
    Ok(map.into_iter().collect())
}
