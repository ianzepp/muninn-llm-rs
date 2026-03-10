//! room:message — streaming LLM-tool ReAct loop.
//!
//! ARCHITECTURE
//! ============
//! `run_actor_loop` implements the ReAct pattern for a single actor:
//!
//! 1. Call `llm:chat` with history + accumulated context + tools.
//! 2. Collect streaming `Item` frames, reconstruct `ContentBlock`s from deltas.
//!    Forward `TextDelta` items upstream as they arrive for low-latency display.
//! 3. Check `stop_reason`:
//!    - `end_turn` / `max_tokens` → emit final reply, return.
//!    - `tool_use` → dispatch each `ToolUse` block, append results to context, loop.
//!    - other → error.
//! 4. After `MAX_TOOL_ROUNDS`, return an error.
//!
//! SECURITY
//! ========
//! Tool calls are checked against the caller-supplied `allowed_prefixes` list.
//! `room:*` calls are blocked (would deadlock). `room:delegate` is intercepted
//! before the allowlist check and runs a nested actor loop inline (Box::pin to
//! break the async recursion cycle).

use tokio::sync::mpsc;
use tracing::{info, warn};

use muninn_kernel::frame::{Frame, Status};
use muninn_kernel::pipe::Caller;
use muninn_kernel::sender::FrameSender;

use crate::client_anthropic::reconstruct_content_blocks;
use crate::error::LlmError;
use crate::room::state::{Actor, HistoryEntry, HistoryKind};
use crate::types::{Content, ContentBlock, ContentDelta, Data, Message, Tool};

const MAX_TOOL_ROUNDS: usize = 20;

// =============================================================================
// ACTOR LOOP
// =============================================================================

/// Run the LLM-tool loop for a single actor, streaming reply deltas upstream.
///
/// Returns the final assembled text reply on success.
pub async fn run_actor_loop(
    caller: &Caller,
    config: &str,
    history: Vec<HistoryEntry>,
    tools: &[Tool],
    allowed_prefixes: &[String],
    room: &str,
    actor_name: &str,
    actors: &[Actor],
    upstream: &FrameSender,
    req_frame: &Frame,
) -> Result<String, LlmError> {
    let mut context: Vec<Message> = Vec::new();

    for round in 0..MAX_TOOL_ROUNDS {
        info!(config = %config, round = round + 1, history = history.len(), context = context.len(), "room: llm round");

        let chat_frame = build_chat_frame(config, &history, &context, tools, room)?;
        let mut stream = caller
            .call(chat_frame)
            .await
            .map_err(|e| LlmError::InternalCall(e.to_string()))?;

        let mut deltas: Vec<ContentDelta> = Vec::new();

        // Stream items from llm:chat, forwarding text deltas upstream.
        loop {
            let Some(frame) = stream.recv().await else {
                break;
            };
            if frame.status == Status::Error {
                let msg = frame
                    .data
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("llm:chat error");
                return Err(LlmError::InternalCall(msg.to_string()));
            }
            if frame.status == Status::Done {
                break;
            }
            if frame.status != Status::Item {
                continue;
            }

            let delta_type = frame
                .data
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Forward text deltas upstream immediately for streaming display.
            if delta_type == "text_delta" {
                let text = frame
                    .data
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !text.is_empty() {
                    let mut d = Data::new();
                    d.insert("type".into(), "text_delta".into());
                    d.insert("actor".into(), actor_name.into());
                    d.insert("room".into(), room.into());
                    d.insert("text".into(), text.clone().into());
                    let _ = upstream.send(req_frame.item(d)).await;

                    deltas.push(ContentDelta::TextDelta(text));
                    continue;
                }
            }

            // Reconstruct other delta types for block assembly.
            let delta = frame_to_delta(&frame.data);
            if let Some(d) = delta {
                deltas.push(d);
            }
        }

        let (blocks, stop_reason_opt) = reconstruct_content_blocks(&deltas);
        let stop_reason = stop_reason_opt.as_deref().unwrap_or("end_turn");

        info!(config = %config, round = round + 1, stop_reason = %stop_reason, blocks = blocks.len(), "room: llm round result");
        emit_broadcasts(caller, room, actor_name, &blocks).await;

        match stop_reason {
            "end_turn" | "max_tokens" => {
                let text = extract_text(&blocks);
                emit_chat(caller, room, actor_name, &text).await;
                emit_reply(upstream, req_frame, room, actor_name, &text).await;
                return Ok(text);
            }
            "tool_use" => {
                // Append assistant turn to context.
                context.push(Message {
                    role: "assistant".to_string(),
                    content: Content::Blocks(blocks.clone()),
                });

                // Dispatch tool calls and collect results.
                let tool_results =
                    dispatch_tools(caller, &blocks, actors, tools, allowed_prefixes, room).await?;
                info!(config = %config, round = round + 1, results = tool_results.len(), "room: tool dispatch done");

                // Append user turn with tool results.
                context.push(Message {
                    role: "user".to_string(),
                    content: Content::Blocks(tool_results),
                });
            }
            other => {
                warn!(stop_reason = %other, "room: unknown stop_reason");
                return Err(LlmError::InternalCall(format!(
                    "unknown stop_reason: {other}"
                )));
            }
        }
    }

    Err(LlmError::InternalCall(format!(
        "tool loop exceeded {MAX_TOOL_ROUNDS} rounds"
    )))
}

// =============================================================================
// TOOL DISPATCH
// =============================================================================

async fn dispatch_tools(
    caller: &Caller,
    blocks: &[ContentBlock],
    actors: &[Actor],
    tools: &[Tool],
    allowed_prefixes: &[String],
    room: &str,
) -> Result<Vec<ContentBlock>, LlmError> {
    let mut results = Vec::new();
    for block in blocks {
        let ContentBlock::ToolUse { id, input, .. } = block else {
            continue;
        };
        let outcome = dispatch_one_tool(caller, input, actors, tools, allowed_prefixes, room).await;
        results.push(ContentBlock::ToolResult {
            tool_use_id: id.clone(),
            content: outcome.content,
            is_error: Some(outcome.is_error),
        });
    }
    Ok(results)
}

async fn dispatch_one_tool(
    caller: &Caller,
    input: &serde_json::Value,
    actors: &[Actor],
    tools: &[Tool],
    allowed_prefixes: &[String],
    room: &str,
) -> ToolOutcome {
    let Some(syscall) = input.get("syscall").and_then(|v| v.as_str()) else {
        return ToolOutcome::error("tool_use input missing 'syscall' field");
    };
    info!(syscall, "room: tool call");

    // Intercept room:delegate before allowlist check.
    if syscall == "room:delegate" {
        return dispatch_delegate(caller, input, actors, tools, allowed_prefixes, room).await;
    }

    // Allow read-only room operations (no deadlock risk).
    let room_read_ok = matches!(syscall, "room:history" | "room:list");

    // Block room:* to prevent deadlock, except read-only above.
    if syscall.starts_with("room:") && !room_read_ok {
        return ToolOutcome::error(format!(
            "forbidden tool syscall (would deadlock): {syscall}"
        ));
    }

    // Allowlist check.
    if !room_read_ok
        && !allowed_prefixes
            .iter()
            .any(|p| syscall.starts_with(p.as_str()))
    {
        return ToolOutcome::error(format!("forbidden tool syscall: {syscall}"));
    }

    let tool_data: Data = match input.get("data").and_then(|v| v.as_object()) {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => Data::new(),
    };

    let mut req = Frame::request(syscall);
    req.data = tool_data;

    let stream = match caller.call(req).await {
        Ok(s) => s,
        Err(e) => return ToolOutcome::error(format!("pipe closed: {e}")),
    };

    let responses = stream.collect().await;
    if let Some(err) = responses.iter().find(|f| f.status == Status::Error) {
        let msg = err
            .data
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("syscall failed");
        return ToolOutcome::error(msg.to_string());
    }

    let items: Vec<String> = responses
        .iter()
        .filter(|f| f.status == Status::Item)
        .filter_map(|f| serde_json::to_string(&f.data).ok())
        .collect();

    let content = if items.is_empty() {
        "ok".to_string()
    } else {
        items.join("\n")
    };
    ToolOutcome {
        content,
        is_error: false,
    }
}

async fn dispatch_delegate(
    caller: &Caller,
    input: &serde_json::Value,
    actors: &[Actor],
    tools: &[Tool],
    allowed_prefixes: &[String],
    room: &str,
) -> ToolOutcome {
    let data = input.get("data").unwrap_or(&serde_json::Value::Null);
    let Some(role) = data.get("role").and_then(|v| v.as_str()) else {
        return ToolOutcome::error("room:delegate missing 'role' in data");
    };
    let Some(prompt) = data.get("prompt").and_then(|v| v.as_str()) else {
        return ToolOutcome::error("room:delegate missing 'prompt' in data");
    };

    let Some(delegate) = actors.iter().find(|a| a.config == role) else {
        return ToolOutcome::error(format!("no actor with role '{role}' in room"));
    };

    info!(role = %role, delegate = %delegate.name, "room: delegating");

    // Minimal single-message history for the delegate.
    let history = vec![HistoryEntry {
        id: 0,
        ts: 0,
        from: "head".to_string(),
        content: prompt.to_string(),
        kind: HistoryKind::User,
    }];

    // Box::pin breaks the async recursion cycle.
    // Delegate replies are discarded (only the return value is used).
    let (tx, _rx) = mpsc::channel(1);
    let sink = FrameSender::new(tx);
    let placeholder_frame = Frame::request("room:delegate");
    match Box::pin(run_actor_loop(
        caller,
        &delegate.config,
        history,
        tools,
        allowed_prefixes,
        room,
        &delegate.name,
        actors,
        &sink,
        &placeholder_frame,
    ))
    .await
    {
        Ok(reply) => ToolOutcome {
            content: reply,
            is_error: false,
        },
        Err(e) => ToolOutcome::error(format!("delegate '{role}' failed: {e}")),
    }
}

// =============================================================================
// HELPERS
// =============================================================================

struct ToolOutcome {
    content: String,
    is_error: bool,
}

impl ToolOutcome {
    fn error(msg: impl Into<String>) -> Self {
        Self {
            content: format!("error: {}", msg.into()),
            is_error: true,
        }
    }
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

async fn emit_broadcasts(caller: &Caller, room: &str, actor_name: &str, blocks: &[ContentBlock]) {
    for block in blocks {
        match block {
            ContentBlock::Thinking { thinking } => {
                let frame = Frame::request("door:thought")
                    .with_from(actor_name)
                    .with_trace(serde_json::json!({ "room": room }))
                    .with_data("room", room.into())
                    .with_data("content", thinking.clone().into());
                if let Err(err) = caller.send(frame).await {
                    info!(error = %err, "room: failed to emit door:thought");
                }
            }
            ContentBlock::ToolUse { input, name, .. } => {
                let syscall = input
                    .get("syscall")
                    .and_then(|value| value.as_str())
                    .unwrap_or(name.as_str());
                let mut frame = Frame::request("door:tool")
                    .with_from(actor_name)
                    .with_trace(serde_json::json!({ "room": room }))
                    .with_data("room", room.into())
                    .with_data("syscall", syscall.into());
                if let Some(args) = input.get("data").cloned() {
                    frame = frame.with_data("args", args);
                }
                if let Err(err) = caller.send(frame).await {
                    info!(error = %err, "room: failed to emit door:tool");
                }
            }
            _ => {}
        }
    }
}

async fn emit_chat(caller: &Caller, room: &str, actor_name: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    let frame = Frame::request("door:chat")
        .with_from(actor_name)
        .with_trace(serde_json::json!({ "room": room }))
        .with_data("room", room.into())
        .with_data("content", text.into());
    if let Err(err) = caller.send(frame).await {
        info!(error = %err, "room: failed to emit door:chat");
    }
}

async fn emit_reply(
    upstream: &FrameSender,
    req_frame: &Frame,
    room: &str,
    actor_name: &str,
    text: &str,
) {
    if text.is_empty() {
        return;
    }
    let mut d = Data::new();
    d.insert("type".into(), "reply".into());
    d.insert("actor".into(), actor_name.into());
    d.insert("room".into(), room.into());
    d.insert("content".into(), text.into());
    let _ = upstream.send(req_frame.item(d)).await;
}

fn frame_to_delta(data: &Data) -> Option<ContentDelta> {
    let delta_type = data.get("type").and_then(|v| v.as_str())?;
    match delta_type {
        "thinking_delta" => {
            let thinking = data
                .get("thinking")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Some(ContentDelta::ThinkingDelta(thinking))
        }
        "tool_use_delta" => {
            let index = data
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(0);
            let id = data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let input_fragment = data
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Some(ContentDelta::ToolUseDelta {
                index,
                id,
                name,
                input_fragment,
            })
        }
        "done" => {
            let stop_reason = data
                .get("stop_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("end_turn")
                .to_string();
            let model = data
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let input_tokens = data
                .get("input_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let output_tokens = data
                .get("output_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            Some(ContentDelta::Done {
                stop_reason,
                model,
                input_tokens,
                output_tokens,
            })
        }
        _ => None,
    }
}

fn build_chat_frame(
    config: &str,
    history: &[HistoryEntry],
    context: &[Message],
    tools: &[Tool],
    room: &str,
) -> Result<Frame, LlmError> {
    let mut data = Data::new();
    data.insert(
        "config".into(),
        serde_json::Value::String(config.to_string()),
    );
    let history_val =
        serde_json::to_value(history).map_err(|e| LlmError::Serialize(e.to_string()))?;
    data.insert("history".into(), history_val);
    if !context.is_empty() {
        let ctx_val =
            serde_json::to_value(context).map_err(|e| LlmError::Serialize(e.to_string()))?;
        data.insert("context".into(), ctx_val);
    }
    if !tools.is_empty() {
        let tools_val =
            serde_json::to_value(tools).map_err(|e| LlmError::Serialize(e.to_string()))?;
        data.insert("tools".into(), tools_val);
    }
    let mut frame = Frame::request("llm:chat");
    frame.data = data;
    frame.trace = Some(serde_json::json!({ "room": room }));
    Ok(frame)
}

#[cfg(test)]
#[path = "message_test.rs"]
mod tests;
