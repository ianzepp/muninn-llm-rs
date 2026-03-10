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
