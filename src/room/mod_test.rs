use std::collections::HashMap;

use serde_json::json;
use tokio::sync::mpsc;

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
fn parse_tools_field_accepts_missing_tools() {
    let data = HashMap::new();
    let tools = parse_tools_field(&data).unwrap();
    assert!(tools.is_empty());
}

#[test]
fn drain_cleanup_removes_closed_worker_from_registry() {
    let (cleanup_tx, cleanup_rx) = mpsc::unbounded_channel();
    let syscall = RoomSyscall {
        workers: Arc::new(Mutex::new(HashMap::new())),
        cleanup_tx,
        cleanup_rx: Arc::new(Mutex::new(cleanup_rx)),
    };
    let (worker_tx, _worker_rx) = mpsc::channel(1);

    syscall
        .workers
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert("general".to_string(), worker_tx);

    syscall.cleanup_tx.send("general".to_string()).unwrap();
    syscall.drain_cleanup();

    assert!(
        syscall
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
}
