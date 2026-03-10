//! RoomSyscall — `room:` prefix handler for the Muninn kernel.
//!
//! ARCHITECTURE
//! ============
//! `RoomSyscall` implements `muninn_kernel::Syscall` for the `"room"` prefix.
//! All room state lives inside this single task (no shared memory, no mutex).
//! Per-room `message` requests are spawned as tasks so concurrent rooms make
//! overlapping LLM calls without blocking the state loop.
//!
//! VERBS
//! =====
//! - `room:join`    — add an actor to a room (creates the room if absent)
//! - `room:part`    — remove an actor from a room (deletes room when empty)
//! - `room:history` — stream history entries for a room
//! - `room:list`    — stream active room names
//! - `room:message` — user message → streaming ReAct loop → reply items

mod handlers;
mod message;
pub mod state;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use muninn_kernel::frame::{ErrorCode, Frame};
use muninn_kernel::pipe::Caller;
use muninn_kernel::sender::FrameSender;
use muninn_kernel::syscall::Syscall;

use crate::config::ConfigFile;
use crate::error::RoomError;
use crate::room::state::{HistoryKind, Room};
use crate::types::{Data, Tool};

// =============================================================================
// CONSTRUCTION
// =============================================================================

/// Kernel syscall handler for the `"room"` prefix.
///
/// Room state is protected by a `Mutex` so `dispatch` (which takes `&self`)
/// can mutate it. Each dispatch holds the lock only for synchronous state
/// updates; the async LLM loop runs outside the lock.
pub struct RoomSyscall {
    rooms: Arc<Mutex<HashMap<String, Room>>>,
}

impl RoomSyscall {
    #[must_use]
    pub fn new(_config: ConfigFile) -> Self {
        Self {
            rooms: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// =============================================================================
// SYSCALL IMPL
// =============================================================================

#[async_trait]
impl Syscall for RoomSyscall {
    fn prefix(&self) -> &'static str {
        "room"
    }

    async fn dispatch(
        &self,
        frame: &Frame,
        tx: &FrameSender,
        caller: &Caller,
        _cancel: CancellationToken,
    ) -> Result<(), Box<dyn ErrorCode + Send>> {
        match frame.verb() {
            "join" => {
                let result = {
                    let mut rooms = lock_rooms(&self.rooms);
                    handlers::handle_join(&mut rooms, frame)
                };
                result.map_err(box_err)?;
                tx.send_done(frame)
                    .await
                    .map_err(|e| box_msg(e.to_string()))?;
                Ok(())
            }
            "part" => {
                let result = {
                    let mut rooms = lock_rooms(&self.rooms);
                    handlers::handle_part(&mut rooms, frame)
                };
                result.map_err(box_err)?;
                tx.send_done(frame)
                    .await
                    .map_err(|e| box_msg(e.to_string()))?;
                Ok(())
            }
            "history" => {
                let items = {
                    let rooms = lock_rooms(&self.rooms);
                    handlers::handle_history(&rooms, frame)
                };
                let items = items.map_err(box_err)?;
                for data in items {
                    tx.send(frame.item(data))
                        .await
                        .map_err(|e| box_msg(e.to_string()))?;
                }
                tx.send_done(frame)
                    .await
                    .map_err(|e| box_msg(e.to_string()))?;
                Ok(())
            }
            "list" => {
                let items = {
                    let rooms = lock_rooms(&self.rooms);
                    handlers::handle_list(&rooms)
                };
                for data in items {
                    tx.send(frame.item(data))
                        .await
                        .map_err(|e| box_msg(e.to_string()))?;
                }
                tx.send_done(frame)
                    .await
                    .map_err(|e| box_msg(e.to_string()))?;
                Ok(())
            }
            "message" => handle_message(self, frame, tx, caller).await,
            other => Err(box_msg(format!("unknown room verb: {other}"))),
        }
    }
}

// =============================================================================
// MESSAGE HANDLER
// =============================================================================

async fn handle_message(
    syscall: &RoomSyscall,
    frame: &Frame,
    tx: &FrameSender,
    caller: &Caller,
) -> Result<(), Box<dyn ErrorCode + Send>> {
    let room_name = str_field(&frame.data, "room").map_err(box_err)?;
    let from = str_field(&frame.data, "from").map_err(box_err)?;
    let content = str_field(&frame.data, "content").map_err(box_err)?;

    let allowed_prefixes: Vec<String> = frame
        .data
        .get("tool_prefixes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let tools: Vec<Tool> = frame
        .data
        .get("tools")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    // Snapshot state needed for the loop — hold lock briefly.
    let (actors, history) = {
        let mut rooms = lock_rooms(&syscall.rooms);
        let room = rooms.entry(room_name.clone()).or_default();

        // Store the user message in history.
        if let Err(e) = room.add_message(&from, &content, HistoryKind::User) {
            return Err(box_err(e));
        }

        let actors = room.actors.clone();
        let history = room.history.clone();
        (actors, history)
    };

    let rooms = Arc::clone(&syscall.rooms);
    let frame_clone = frame.clone();
    let caller_clone = caller.clone();
    let tx_clone = tx.clone();

    // Spawn actors concurrently, collecting replies.
    let mut join_handles = Vec::new();
    for actor in &actors {
        let actor_name = actor.name.clone();
        let actor_config = actor.config.clone();
        let room_name_clone = room_name.clone();
        let history_clone = history.clone();
        let tools_clone = tools.clone();
        let allowed_clone = allowed_prefixes.clone();
        let actors_clone = actors.clone();
        let caller_c = caller_clone.clone();
        let tx_c = tx_clone.clone();
        let frame_c = frame_clone.clone();

        let handle = tokio::spawn(async move {
            message::run_actor_loop(
                &caller_c,
                &actor_config,
                history_clone,
                &tools_clone,
                &allowed_clone,
                &room_name_clone,
                &actor_name,
                &actors_clone,
                &tx_c,
                &frame_c,
            )
            .await
            .map(|reply| (actor_name, reply))
        });
        join_handles.push(handle);
    }

    // Collect replies and store in history.
    for handle in join_handles {
        let result = handle.await.map_err(|e| box_msg(e.to_string()))?;
        match result {
            Ok((actor_name, reply)) => {
                let mut rooms = lock_rooms(&rooms);
                if let Some(room) = rooms.get_mut(&room_name) {
                    let _ = room.add_message(&actor_name, &reply, HistoryKind::Assistant);
                }
            }
            Err(e) => {
                warn!(error = %e, "room: actor loop error");
            }
        }
    }

    tx_clone
        .send_done(&frame_clone)
        .await
        .map_err(|e| box_msg(e.to_string()))?;
    Ok(())
}

// =============================================================================
// HELPERS
// =============================================================================

fn lock_rooms(
    rooms: &Arc<Mutex<HashMap<String, Room>>>,
) -> std::sync::MutexGuard<'_, HashMap<String, Room>> {
    rooms
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn str_field(data: &Data, key: &str) -> Result<String, RoomError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize(format!("missing '{key}' field")))
}

fn box_err(e: impl ErrorCode + Send + 'static) -> Box<dyn ErrorCode + Send> {
    Box::new(e)
}

fn box_msg(msg: String) -> Box<dyn ErrorCode + Send> {
    Box::new(crate::error::LlmError::InternalCall(msg))
}
