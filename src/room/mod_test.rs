use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::json;

use super::*;

#[test]
fn parse_tools_field_rejects_invalid_schema() {
    let mut data = Data::new();
    data.insert("tools".into(), json!({"name":"not-an-array"}));

    let err = parse_tools_field(&data).unwrap_err();
    assert!(matches!(err, RoomError::Deserialize(_)));
    assert!(err.to_string().contains("invalid 'tools' field"));
}

#[test]
fn room_message_guard_rejects_same_room_overlap() {
    let in_flight = Arc::new(Mutex::new(HashSet::new()));
    let guard = RoomMessageGuard::acquire(Arc::clone(&in_flight), "general").unwrap();

    let Err(err) = RoomMessageGuard::acquire(Arc::clone(&in_flight), "general") else {
        panic!("second guard acquisition should fail");
    };
    assert!(matches!(
        err,
        RoomError::RoomBusy { room } if room == "general"
    ));

    drop(guard);

    let guard = RoomMessageGuard::acquire(Arc::clone(&in_flight), "general").unwrap();
    drop(guard);
    assert!(
        in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}

#[test]
fn parse_tools_field_accepts_missing_tools() {
    let data = HashMap::new();
    let tools = parse_tools_field(&data).unwrap();
    assert!(tools.is_empty());
}
