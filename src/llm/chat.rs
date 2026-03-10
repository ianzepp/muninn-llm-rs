//! llm:chat handler — emit provider deltas as Item frames.
//!
//! ARCHITECTURE
//! ============
//! Receives a `ChatRequest` from frame data, selects the configured provider
//! client, builds the system prompt, converts history to provider messages,
//! collects deltas from the provider, and emits each as an `Item` frame.
//!
//! FRAME STREAM
//! ============
//! Item { type: "text_delta",     text }
//! Item { type: "thinking_delta", thinking }
//! Item { type: "tool_use_delta", id, name, input }
//! Item { type: "done", stop_reason, model, input_tokens, output_tokens }
//! Done (terminal)

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use tracing::info;

use muninn_kernel::frame::Frame;
use muninn_kernel::sender::FrameSender;

use crate::client_anthropic::AnthropicClient;
use crate::client_openai::OpenAiClient;
use crate::config::{ConfigFile, LlmProfile, resolve_api_key};
use crate::error::LlmError;
use crate::prompt_bundle::{PromptContext, build_system_prompt};
use crate::types::{ChatRequest, Content, ContentDelta, Data, Message, from_data};

// =============================================================================
// PROVIDER CLIENT ENUM
// =============================================================================

/// A provider-specific HTTP client, unified behind a single streaming interface.
///
/// WHY an enum rather than a trait object: `stream_chat` takes a generic
/// `FnMut` callback with an associated `Future`, which cannot be expressed as
/// a `dyn Trait` without boxing every future. The enum avoids that overhead
/// while still keeping provider dispatch in one place.
pub(crate) enum ProviderClient {
    Anthropic(AnthropicClient),
    OpenAi(OpenAiClient),
}

impl ProviderClient {
    /// Construct a `ProviderClient` from a config profile.
    ///
    /// Resolves the API key and validates `openai_api` mode eagerly so that
    /// misconfiguration fails at startup rather than on the first request.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError`] if the API key is missing, the provider name is
    /// unknown, or the OpenAI API mode is unsupported.
    pub(crate) fn build(profile: &LlmProfile) -> Result<Self, LlmError> {
        let api_key = resolve_api_key(profile)?;
        match profile.provider.as_str() {
            "anthropic" => Ok(Self::Anthropic(AnthropicClient::new(api_key)?)),
            "openai" => {
                let api_mode = profile.openai_api.as_deref().unwrap_or("chat_completions");
                if api_mode != "chat_completions" {
                    return Err(LlmError::ConfigParse(format!(
                        "unsupported openai_api: {api_mode}"
                    )));
                }
                Ok(Self::OpenAi(OpenAiClient::new(
                    api_key,
                    profile.openai_base_url.as_deref(),
                )?))
            }
            other => Err(LlmError::ConfigParse(format!(
                "unsupported provider: {other}"
            ))),
        }
    }

    /// Stream a chat completion, invoking `on_delta` for each decoded delta.
    ///
    /// Delegates to the concrete provider client. The `on_delta` callback
    /// receives each [`ContentDelta`] as it arrives from the provider stream,
    /// allowing the caller to forward fragments upstream without buffering.
    pub(crate) async fn stream_chat<F, Fut>(
        &self,
        model: &str,
        max_tokens: u32,
        system_prompt: &str,
        messages: &[Message],
        tools: Option<&[crate::types::Tool]>,
        on_delta: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(ContentDelta) -> Fut,
        Fut: Future<Output = Result<(), LlmError>>,
    {
        match self {
            Self::Anthropic(client) => {
                client
                    .stream_chat(model, max_tokens, system_prompt, messages, tools, on_delta)
                    .await
            }
            Self::OpenAi(client) => {
                client
                    .stream_chat(model, max_tokens, system_prompt, messages, tools, on_delta)
                    .await
            }
        }
    }
}

/// Build one `ProviderClient` per named config profile, returning them as a map.
///
/// Called once at `LlmSyscall` construction time. Building all clients eagerly
/// means API key errors are surfaced before the first request, not mid-flight.
pub(crate) fn build_clients(
    config: &ConfigFile,
) -> Result<HashMap<String, ProviderClient>, LlmError> {
    let mut clients = HashMap::new();
    for (name, profile) in &config.configs {
        clients.insert(name.clone(), ProviderClient::build(profile)?);
    }
    Ok(clients)
}

// =============================================================================
// HANDLE CHAT
// =============================================================================

/// Serve a single `llm:chat` request, emitting provider deltas as `Item` frames.
///
/// This function runs inside a spawned task (see `LlmSyscall::dispatch`) and
/// owns the full lifecycle of the provider stream for one request. It sends
/// the terminal `Done` or `Error` frame before returning.
///
/// # Frame Stream Emitted
///
/// ```text
/// Item { type: "text_delta",     text }           — one per text fragment
/// Item { type: "thinking_delta", thinking }       — one per thinking fragment
/// Item { type: "tool_use_delta", id, name, input } — one per tool call fragment
/// Item { type: "done", stop_reason, model, ... }  — once at stream end
/// Done                                             — terminal frame
/// ```
pub(crate) async fn handle_chat(
    frame: Frame,
    config: Arc<ConfigFile>,
    clients: Arc<HashMap<String, ProviderClient>>,
    tx: &FrameSender,
) {
    // PHASE 1: DESERIALIZE AND RESOLVE CONFIG
    // Parse the request payload and look up the named profile and its client.
    // Both lookups must succeed before any network activity begins.
    let req: ChatRequest = match from_data(&frame.data) {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send_error_from(&frame, &e).await;
            return;
        }
    };

    let Some(profile) = config.configs.get(&req.config) else {
        let _ = tx
            .send_error(&frame, format!("unknown config: {}", req.config))
            .await;
        return;
    };

    let Some(client) = clients.get(&req.config) else {
        let _ = tx
            .send_error(&frame, format!("no client for config: {}", req.config))
            .await;
        return;
    };

    // PHASE 2: BUILD SYSTEM PROMPT AND MESSAGE LIST
    // Assemble the cacheable system prompt from the profile and call context,
    // then flatten history entries and in-progress context into a message list.
    // The room name is read from the frame trace rather than the data payload
    // so it is available even when the caller does not include it in data.
    let room = frame
        .trace
        .as_ref()
        .and_then(|t| t.get("room"))
        .and_then(|r| r.as_str())
        .unwrap_or("");

    let system_prompt = build_system_prompt(&PromptContext {
        config: &req.config,
        self_prompt: &profile.self_prompt,
        tools: req.tools.as_deref(),
        room,
        description: "",
        notes: "",
        memory: req.memory.as_deref().unwrap_or(""),
        traits: &config.traits,
    });

    // WHY extend with context after history: context holds the in-progress
    // tool loop turns for the current request. They must appear after the
    // committed history so the model sees them as the most recent turns.
    let mut messages = history_to_messages(&req.history);
    messages.extend(req.context);

    let tools = req.tools.as_deref();
    let model = profile.model.clone();
    let max_tokens = profile.max_tokens;

    info!(config = %req.config, room = %room, history = req.history.len(), "llm: chat start");

    // PHASE 3: STREAM PROVIDER RESPONSE
    // Each delta is forwarded upstream as an Item frame immediately so the
    // caller can display text as it arrives without waiting for the full turn.
    let stream_result = client
        .stream_chat(
            &model,
            max_tokens,
            &system_prompt,
            &messages,
            tools,
            |delta| {
                let item_frame = frame.clone();
                async move {
                    let data = delta_to_data(&delta);
                    tx.send(item_frame.item(data))
                        .await
                        .map_err(|e| LlmError::PipeSend(e.to_string()))
                }
            },
        )
        .await;

    match stream_result {
        Ok(()) => {}
        Err(e) => {
            let _ = tx.send_error_from(&frame, &e).await;
            return;
        }
    }

    let _ = tx.send_done(&frame).await;
    info!(config = %req.config, "llm: chat done");
}

// =============================================================================
// DELTA → FRAME DATA
// =============================================================================

/// Convert a [`ContentDelta`] into the flat `Data` map emitted as an `Item` frame.
///
/// The `type` key discriminates the delta kind for downstream consumers, matching
/// the frame stream contract documented on [`handle_chat`].
fn delta_to_data(delta: &ContentDelta) -> Data {
    let mut d = Data::new();
    match delta {
        ContentDelta::TextDelta(text) => {
            d.insert("type".into(), "text_delta".into());
            d.insert("text".into(), text.clone().into());
        }
        ContentDelta::ThinkingDelta(thinking) => {
            d.insert("type".into(), "thinking_delta".into());
            d.insert("thinking".into(), thinking.clone().into());
        }
        ContentDelta::ToolUseDelta {
            index,
            id,
            name,
            input_fragment,
        } => {
            d.insert("type".into(), "tool_use_delta".into());
            d.insert("index".into(), (*index).into());
            d.insert("id".into(), id.clone().into());
            d.insert("name".into(), name.clone().into());
            d.insert("input".into(), input_fragment.clone().into());
        }
        ContentDelta::Done {
            stop_reason,
            model,
            input_tokens,
            output_tokens,
        } => {
            d.insert("type".into(), "done".into());
            d.insert("stop_reason".into(), stop_reason.clone().into());
            d.insert("model".into(), model.clone().into());
            d.insert("input_tokens".into(), (*input_tokens).into());
            d.insert("output_tokens".into(), (*output_tokens).into());
        }
    }
    d
}

// =============================================================================
// HISTORY → MESSAGES
// =============================================================================

/// Convert room history entries into provider-neutral [`Message`] values.
///
/// WHY filter to User/Assistant only: `HistoryKind` may be extended with
/// system-level or metadata entries in the future. Only `User` and `Assistant`
/// map to valid LLM message roles; other kinds carry no information the model
/// should see and are silently dropped here rather than causing downstream
/// serialization failures.
pub fn history_to_messages(history: &[crate::room::state::HistoryEntry]) -> Vec<Message> {
    use crate::room::state::HistoryKind;

    let relevant: Vec<_> = history
        .iter()
        .filter(|e| matches!(e.kind, HistoryKind::User | HistoryKind::Assistant))
        .collect();

    let mut messages = Vec::with_capacity(relevant.len());

    for entry in &relevant {
        messages.push(Message {
            role: entry.kind.as_str().to_string(),
            content: Content::Text(entry.content.clone()),
        });
    }

    messages
}

#[cfg(test)]
#[path = "chat_test.rs"]
mod tests;
