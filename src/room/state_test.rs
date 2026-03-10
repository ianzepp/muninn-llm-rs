use super::*;

#[test]
fn add_message_increments_id() {
    let mut room = Room::new();
    let id1 = room
        .add_message("user", "hello", HistoryKind::User)
        .unwrap();
    let id2 = room
        .add_message("bot", "hi", HistoryKind::Assistant)
        .unwrap();
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
    assert_eq!(room.history.len(), 2);
}

#[test]
fn trim_history_removes_oldest() {
    let mut room = Room::new();
    for i in 0..5 {
        room.add_message("user", format!("msg {i}"), HistoryKind::User)
            .unwrap();
    }
    room.trim_history(3);
    assert_eq!(room.history.len(), 3);
    assert_eq!(room.history[0].content, "msg 2");
}

#[test]
fn render_recent_tool_outcomes_includes_status_and_error_code() {
    let mut room = Room::new();
    room.add_tool_outcome(ToolOutcomeRecord {
        actor: "bot".to_string(),
        syscall: "exec:run".to_string(),
        ok: false,
        summary: "command not found".to_string(),
        error_code: Some("E_NOT_FOUND".to_string()),
        ts: 1,
        turn_id: 7,
    });
    room.add_tool_outcome(ToolOutcomeRecord {
        actor: "bot".to_string(),
        syscall: "vfs:read".to_string(),
        ok: true,
        summary: "loaded README".to_string(),
        error_code: None,
        ts: 2,
        turn_id: 8,
    });

    let rendered = room.render_recent_tool_outcomes(8);
    assert!(rendered.contains("turn 7"));
    assert!(rendered.contains("exec:run"));
    assert!(rendered.contains("error"));
    assert!(rendered.contains("E_NOT_FOUND"));
    assert!(rendered.contains("vfs:read"));
    assert!(rendered.contains("ok"));
}
