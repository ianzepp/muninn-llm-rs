//! Pure room state mutation handlers used by the worker for verb dispatch.
//!
//! DESIGN
//! ======
//! These functions operate on a `&mut HashMap<String, Room>` (or `&HashMap`)
//! rather than receiving `WorkerRequest` directly. The separation keeps room
//! state logic synchronous and easily testable without the async worker harness.
//! The worker is responsible for sending response frames; these functions return
//! `Result` values or computed data for the worker to act on.
//!
//! NOTE: These handlers are currently unused in favour of the inline handler
//! methods on `RoomWorker` in `worker.rs`. They are retained for potential use
//! by a future stateless or shared-state room variant.

use std::collections::HashMap;

use tracing::info;

use muninn_kernel::frame::Frame;

use crate::error::RoomError;
use crate::room::state::{Actor, Room};
use crate::types::Data;

// =============================================================================
// JOIN
// =============================================================================

/// Add an actor to a room, creating the room if it does not exist.
///
/// The room is created with `entry().or_default()` so no explicit creation
/// step is required — `room:join` is the implicit room constructor.
///
/// # Errors
///
/// Returns [`RoomError::ActorAlreadyJoined`] if the actor name is already
/// present in the room.
pub fn handle_join(rooms: &mut HashMap<String, Room>, frame: &Frame) -> Result<(), RoomError> {
    let room_name = room_name(frame)?;
    let actor_name = str_field(&frame.data, "actor_name")?;
    let config = str_field(&frame.data, "config")?;

    let room = rooms.entry(room_name.clone()).or_default();

    if room.actors.iter().any(|a| a.name == actor_name) {
        return Err(RoomError::ActorAlreadyJoined {
            room: room_name,
            name: actor_name,
        });
    }

    room.actors.push(Actor {
        name: actor_name.clone(),
        config,
    });
    info!(room = %room_name, actor = %actor_name, "room: actor joined");
    Ok(())
}

// =============================================================================
// PART
// =============================================================================

/// Remove an actor from a room, destroying the room if it becomes empty.
///
/// Returns `true` if the room was destroyed (the worker should exit after
/// sending the done frame). Returns `false` if other actors remain.
///
/// # Errors
///
/// Returns [`RoomError::RoomNotFound`] if the room does not exist, or
/// [`RoomError::ActorNotFound`] if the actor is not in the room.
pub fn handle_part(rooms: &mut HashMap<String, Room>, frame: &Frame) -> Result<bool, RoomError> {
    let room_name = room_name(frame)?;
    let actor_name = str_field(&frame.data, "actor_name")?;

    let Some(room) = rooms.get_mut(&room_name) else {
        return Err(RoomError::RoomNotFound { room: room_name });
    };

    let before = room.actors.len();
    room.actors.retain(|a| a.name != actor_name);
    if room.actors.len() == before {
        return Err(RoomError::ActorNotFound {
            room: room_name,
            name: actor_name,
        });
    }

    info!(room = %room_name, actor = %actor_name, "room: actor parted");
    let empty = room.actors.is_empty();
    if empty {
        rooms.remove(&room_name);
    }
    Ok(empty)
}

// =============================================================================
// HISTORY
// =============================================================================

/// Return serialized history entries for a room, optionally limited to the
/// most recent `N` entries.
///
/// WHY rev/take/rev for limit: `history` is ordered oldest-first. Taking from
/// the tail (most recent) and then reversing back to chronological order gives
/// the `N` most recent entries in their original order. A simple `take(N)` from
/// the front would return the oldest entries instead.
///
/// # Errors
///
/// Returns [`RoomError::RoomNotFound`] if the room does not exist.
pub fn handle_history(
    rooms: &HashMap<String, Room>,
    frame: &Frame,
) -> Result<Vec<Data>, RoomError> {
    let room_name = room_name(frame)?;

    let Some(room) = rooms.get(&room_name) else {
        return Err(RoomError::RoomNotFound { room: room_name });
    };

    let limit = frame
        .data
        .get("limit")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());

    let entries: Vec<&crate::room::state::HistoryEntry> = match limit {
        Some(n) => room
            .history
            .iter()
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect(),
        None => room.history.iter().collect(),
    };

    let mut items = Vec::new();
    for entry in entries {
        let mut d = Data::new();
        d.insert("id".into(), entry.id.into());
        d.insert("ts".into(), entry.ts.into());
        d.insert("from".into(), entry.from.clone().into());
        d.insert("content".into(), entry.content.clone().into());
        d.insert("kind".into(), entry.kind.as_str().into());
        items.push(d);
    }
    Ok(items)
}

// =============================================================================
// LIST
// =============================================================================

/// Return a list of active room names, each as a `Data` map with a `"room"` key.
pub fn handle_list(rooms: &HashMap<String, Room>) -> Vec<Data> {
    rooms
        .keys()
        .map(|name| {
            let mut d = Data::new();
            d.insert("room".into(), name.clone().into());
            d
        })
        .collect()
}

// =============================================================================
// HELPERS
// =============================================================================

/// Extract the required `"room"` field from frame data.
fn room_name(frame: &Frame) -> Result<String, RoomError> {
    frame
        .data
        .get("room")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize("missing 'room' field".into()))
}

/// Extract a required string field from a `Data` map.
fn str_field(data: &Data, key: &str) -> Result<String, RoomError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize(format!("missing '{key}' field")))
}
