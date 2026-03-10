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
///
/// The `input_schema` is a JSON Schema object describing the tool's input.
/// Both Anthropic (`input_schema`) and OpenAI (`parameters`) accept the same
/// JSON Schema shape, so no conversion is needed between providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// =============================================================================
// MESSAGE
// =============================================================================

/// A single message in a conversation — role plus content.
///
/// `role` is one of `"user"`, `"assistant"`, or `"system"`. The `content`
/// field is either plain text (for simple turns) or a block array (for turns
/// containing tool calls or tool results).
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
/// WHY history + context: `history` is the room's flat conversation log
/// (user and assistant turns stored after each completed `room:message`).
/// `context` carries the in-progress tool loop turns for the current round —
/// assistant responses with `tool_use` blocks and their `tool_result` replies —
/// which the room accumulates between ReAct iterations and has not yet committed
/// to history. Separating the two allows the LLM to see both the stable history
/// and the live multi-step context without conflating them.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Profile name from `ConfigFile` selecting model, provider, and key.
    pub config: String,
    /// Committed room history (user and assistant turns).
    pub history: Vec<HistoryEntry>,
    /// In-progress tool loop turns for the current `room:message` request.
    /// Empty on the first round; grows as tool calls are dispatched and
    /// their results appended before subsequent LLM rounds.
    #[serde(default)]
    pub context: Vec<Message>,
    /// Optional long-term memory text injected into the system prompt.
    /// This crate never populates this; the caller (room worker) may provide it.
    #[serde(default)]
    pub memory: Option<String>,
    /// Tool definitions made available to the LLM for this request.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

/// Final assembled response from `llm:chat`.
///
/// This is a convenience type for callers that want a single structured
/// result rather than consuming individual `Item` frames. The room worker
/// does not use this type — it reads deltas directly from the frame stream
/// to enable low-latency forwarding.
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

/// The `Frame.data` map type — a flat `String → Value` map.
///
/// WHY `HashMap<String, Value>` rather than a typed struct: `Frame` is a
/// kernel-level primitive shared across all syscalls. The kernel cannot know
/// every payload shape, so it uses a generic map. Each syscall handler
/// deserializes the relevant fields it needs using [`from_data`]/[`to_data`].
pub type Data = HashMap<String, Value>;

/// Deserialize a typed request from a `Frame.data` map.
///
/// Reconstitutes the map as a `serde_json::Value::Object` so serde's derived
/// `Deserialize` impls can be used directly, avoiding manual field extraction.
///
/// # Errors
///
/// Returns [`LlmError::Deserialize`] if the map cannot be deserialized into `T`.
pub fn from_data<T: serde::de::DeserializeOwned>(data: &Data) -> Result<T, LlmError> {
    let map: serde_json::Map<String, Value> =
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    serde_json::from_value(Value::Object(map)).map_err(|e| LlmError::Deserialize(e.to_string()))
}

/// Serialize a typed value into a `Frame.data` map.
///
/// The value must serialize as a JSON object (i.e., it must be a struct or
/// map). Primitive values and arrays are rejected with [`LlmError::Serialize`].
///
/// # Errors
///
/// Returns [`LlmError::Serialize`] if `value` cannot be serialized or does not
/// produce a JSON object at the top level.
pub fn to_data<T: Serialize>(value: &T) -> Result<Data, LlmError> {
    let v = serde_json::to_value(value).map_err(|e| LlmError::Serialize(e.to_string()))?;
    let Value::Object(map) = v else {
        return Err(LlmError::Serialize("expected object".into()));
    };
    Ok(map.into_iter().collect())
}
