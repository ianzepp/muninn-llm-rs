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
use std::sync::Arc;

use tracing::info;

use muninn_kernel::frame::Frame;
use muninn_kernel::sender::FrameSender;

use crate::client_anthropic::AnthropicClient;
use crate::client_openai::OpenAiClient;
use crate::config::{ConfigFile, LlmProfile, resolve_api_key};
use crate::error::LlmError;
use crate::prompt_bundle::{PromptContext, build_system_prompt};
use crate::types::{Content, ContentDelta, Data, Message, from_data, ChatRequest};

// =============================================================================
// PROVIDER CLIENT ENUM
// =============================================================================

pub(crate) enum ProviderClient {
    Anthropic(AnthropicClient),
    OpenAi(OpenAiClient),
}

impl ProviderClient {
    pub(crate) fn build(profile: &LlmProfile) -> Result<Self, LlmError> {
        let api_key = resolve_api_key(profile)?;
        match profile.provider.as_str() {
            "anthropic" => Ok(Self::Anthropic(AnthropicClient::new(api_key)?)),
            "openai" => Ok(Self::OpenAi(OpenAiClient::new(api_key, profile.openai_base_url.as_deref())?)),
            other => Err(LlmError::ConfigParse(format!("unsupported provider: {other}"))),
        }
    }
}

pub(crate) fn build_clients(config: &ConfigFile) -> Result<HashMap<String, ProviderClient>, LlmError> {
    let mut clients = HashMap::new();
    for (name, profile) in &config.configs {
        clients.insert(name.clone(), ProviderClient::build(profile)?);
    }
    Ok(clients)
}

// =============================================================================
// HANDLE CHAT
// =============================================================================

pub(crate) async fn handle_chat(
    frame: Frame,
    config: Arc<ConfigFile>,
    clients: Arc<HashMap<String, ProviderClient>>,
    tx: &FrameSender,
) {
    let req: ChatRequest = match from_data(&frame.data) {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send_error_from(&frame, &e).await;
            return;
        }
    };

    let Some(profile) = config.configs.get(&req.config) else {
        let _ = tx.send_error(&frame, format!("unknown config: {}", req.config)).await;
        return;
    };

    let Some(client) = clients.get(&req.config) else {
        let _ = tx.send_error(&frame, format!("no client for config: {}", req.config)).await;
        return;
    };

    let room = frame.trace.as_ref()
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
        memory: "",
        traits: &config.traits,
    });

    let mut messages = history_to_messages(&req.history);
    messages.extend(req.context);

    let tools = req.tools.as_deref();
    let model = profile.model.clone();
    let max_tokens = profile.max_tokens;

    info!(config = %req.config, room = %room, history = req.history.len(), "llm: chat start");

    let deltas_result = match client {
        ProviderClient::Anthropic(c) => c.stream_chat(&model, max_tokens, &system_prompt, &messages, tools).await,
        ProviderClient::OpenAi(c) => c.stream_chat(&model, max_tokens, &system_prompt, &messages, tools).await,
    };

    let deltas = match deltas_result {
        Ok(d) => d,
        Err(e) => {
            let _ = tx.send_error_from(&frame, &e).await;
            return;
        }
    };

    for delta in &deltas {
        if let Some(data) = delta_to_data(delta) {
            if tx.send(frame.item(data)).await.is_err() {
                return;
            }
        }
    }

    let _ = tx.send_done(&frame).await;
    info!(config = %req.config, "llm: chat done");
}

// =============================================================================
// DELTA → FRAME DATA
// =============================================================================

fn delta_to_data(delta: &ContentDelta) -> Option<Data> {
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
        ContentDelta::ToolUseDelta { id, name, input_fragment } => {
            d.insert("type".into(), "tool_use_delta".into());
            d.insert("id".into(), id.clone().into());
            d.insert("name".into(), name.clone().into());
            d.insert("input".into(), input_fragment.clone().into());
        }
        ContentDelta::Done { stop_reason, model, input_tokens, output_tokens } => {
            d.insert("type".into(), "done".into());
            d.insert("stop_reason".into(), stop_reason.clone().into());
            d.insert("model".into(), model.clone().into());
            d.insert("input_tokens".into(), (*input_tokens).into());
            d.insert("output_tokens".into(), (*output_tokens).into());
        }
    }
    Some(d)
}

// =============================================================================
// HISTORY → MESSAGES
// =============================================================================

const CHRONO_GAP_THRESHOLD_SECS: i64 = 600;

/// Convert room history entries to provider-neutral messages with chrono markers.
pub fn history_to_messages(history: &[crate::room::state::HistoryEntry]) -> Vec<Message> {
    use crate::room::state::HistoryKind;

    let relevant: Vec<_> = history
        .iter()
        .filter(|e| matches!(e.kind, HistoryKind::User | HistoryKind::Assistant))
        .collect();

    let mut messages = Vec::with_capacity(relevant.len() + relevant.len() / 4 + 1);
    let mut prev_ts: Option<i64> = None;

    for entry in &relevant {
        if let Some(prev) = prev_ts {
            let gap = entry.ts.saturating_sub(prev);
            if gap >= CHRONO_GAP_THRESHOLD_SECS {
                messages.push(chrono_gap_message(gap));
            }
        }
        messages.push(Message {
            role: entry.kind.as_str().to_string(),
            content: Content::Text(entry.content.clone()),
        });
        prev_ts = Some(entry.ts);
    }

    messages.push(chrono_current_message());
    messages
}

fn chrono_gap_message(gap_secs: i64) -> Message {
    let label = format_duration(gap_secs);
    Message { role: "user".to_string(), content: Content::Text(format!("<chrono type=\"gap\">{label}</chrono>")) }
}

fn chrono_current_message() -> Message {
    use chrono::{DateTime, Utc};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    #[allow(clippy::cast_possible_wrap)]
    let formatted = DateTime::<Utc>::from_timestamp(now as i64, 0)
        .map_or_else(|| "(unknown)".to_string(), |dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string());
    Message {
        role: "user".to_string(),
        content: Content::Text(format!("<chrono type=\"current\">{formatted}</chrono>")),
    }
}

fn format_duration(secs: i64) -> String {
    if secs < 60 { return format!("{secs}s"); }
    let mins = secs / 60;
    if mins < 60 { return format!("{mins}min"); }
    let hours = mins / 60;
    let rem = mins % 60;
    if rem == 0 { format!("{hours}h") } else { format!("{hours}h {rem}min") }
}
