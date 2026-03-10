//! Room verb handlers: join, part, history, list.

use std::collections::HashMap;

use tracing::info;

use muninn_kernel::frame::Frame;

use crate::error::RoomError;
use crate::room::state::{Actor, Room};
use crate::types::Data;

// =============================================================================
// JOIN
// =============================================================================

pub fn handle_join(
    rooms: &mut HashMap<String, Room>,
    frame: &Frame,
) -> Result<(), RoomError> {
    let room_name = room_name(frame)?;
    let actor_name = str_field(&frame.data, "actor_name")?;
    let config = str_field(&frame.data, "config")?;

    let room = rooms.entry(room_name.clone()).or_insert_with(Room::new);

    if room.actors.iter().any(|a| a.name == actor_name) {
        return Err(RoomError::ActorAlreadyJoined { room: room_name, name: actor_name });
    }

    room.actors.push(Actor { name: actor_name.clone(), config });
    info!(room = %room_name, actor = %actor_name, "room: actor joined");
    Ok(())
}

// =============================================================================
// PART
// =============================================================================

pub fn handle_part(
    rooms: &mut HashMap<String, Room>,
    frame: &Frame,
) -> Result<bool, RoomError> {
    let room_name = room_name(frame)?;
    let actor_name = str_field(&frame.data, "actor_name")?;

    let Some(room) = rooms.get_mut(&room_name) else {
        return Err(RoomError::RoomNotFound { room: room_name });
    };

    let before = room.actors.len();
    room.actors.retain(|a| a.name != actor_name);
    if room.actors.len() == before {
        return Err(RoomError::ActorNotFound { room: room_name, name: actor_name });
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

pub fn handle_history<'a>(
    rooms: &'a HashMap<String, Room>,
    frame: &Frame,
) -> Result<Vec<Data>, RoomError> {
    let room_name = room_name(frame)?;

    let Some(room) = rooms.get(&room_name) else {
        return Err(RoomError::RoomNotFound { room: room_name });
    };

    let limit = frame.data.get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);

    let entries: Vec<&crate::room::state::HistoryEntry> = match limit {
        Some(n) => room.history.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect(),
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

fn room_name(frame: &Frame) -> Result<String, RoomError> {
    frame.data
        .get("room")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize("missing 'room' field".into()))
}

fn str_field(data: &Data, key: &str) -> Result<String, RoomError> {
    data.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RoomError::Deserialize(format!("missing '{key}' field")))
}
