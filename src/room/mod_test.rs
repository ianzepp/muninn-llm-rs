use std::time::Duration;

use std::collections::HashMap;

use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use muninn_kernel::frame::{ErrorCode, Frame, Status};
use muninn_kernel::pipe::{PipeEnd, pipe};
use muninn_kernel::sender::FrameSender;
use muninn_kernel::syscall::Syscall;

use crate::config::ConfigFile;

use super::*;

#[derive(Debug)]
struct TestError(&'static str);

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl ErrorCode for TestError {
    fn error_code(&self) -> &'static str {
        "E_TEST_EXEC"
    }
}

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

#[tokio::test]
async fn room_message_next_turn_receives_compact_prior_tool_failure_memory() {
    let (mut room_end, mut test_end) = pipe(64);
    let syscall = RoomSyscall::new(ConfigFile {
        active: "default".to_string(),
        configs: HashMap::default(),
        traits: Vec::new(),
    });

    let dispatch_task = tokio::spawn(async move {
        let sender = FrameSender::new(room_end.sender());
        let caller = room_end.caller();

        while let Some(frame) = room_end.recv().await {
            if frame.status != Status::Request {
                continue;
            }

            let _ = syscall
                .dispatch(&frame, &sender, &caller, CancellationToken::new())
                .await;
        }
    });

    let join = request_with_data(
        "room:join",
        serde_json::json!({
            "room": "general",
            "actor_name": "bot-1",
            "config": "default"
        }),
    );
    send_to_room(&test_end, join.clone()).await;
    let join_done = recv_frame(&mut test_end).await;
    assert_eq!(join_done.parent_id, Some(join.id));
    assert_eq!(join_done.status, Status::Done);

    let turn1 = request_with_data(
        "room:message",
        serde_json::json!({
            "room": "general",
            "from": "alice",
            "content": "try the command",
            "tool_prefixes": ["exec:"]
        }),
    );
    send_to_room(&test_end, turn1.clone()).await;

    let llm_req1 = recv_request(&mut test_end, "llm:chat").await;
    assert!(!llm_req1.data.contains_key("memory"));
    send_tool_use_round(&test_end, &llm_req1).await;

    let exec_req = recv_request(&mut test_end, "exec:run").await;
    send_error_response(&test_end, &exec_req, &TestError("permission denied")).await;

    let llm_req2 = recv_request(&mut test_end, "llm:chat").await;
    assert!(!llm_req2.data.contains_key("memory"));
    send_text_round(&test_end, &llm_req2, "The tool failed previously.").await;

    let turn1_frames = collect_response_frames(&mut test_end, turn1.id).await;
    assert!(turn1_frames.iter().any(|frame| {
        frame.status == Status::Item
            && frame.data.get("type").and_then(|value| value.as_str()) == Some("reply")
            && frame.data.get("content").and_then(|value| value.as_str())
                == Some("The tool failed previously.")
    }));
    assert_eq!(
        turn1_frames.last().map(|frame| frame.status),
        Some(Status::Done)
    );

    let turn2 = request_with_data(
        "room:message",
        serde_json::json!({
            "room": "general",
            "from": "alice",
            "content": "what happened before?",
            "tool_prefixes": ["exec:"]
        }),
    );
    send_to_room(&test_end, turn2.clone()).await;

    let llm_req3 = recv_request(&mut test_end, "llm:chat").await;
    let memory = llm_req3
        .data
        .get("memory")
        .and_then(|value| value.as_str())
        .expect("turn 2 should include compact tool memory");
    assert!(memory.contains("exec:run"));
    assert!(memory.contains("error"));
    assert!(memory.contains("permission denied"));
    assert!(memory.contains("E_TEST_EXEC"));

    let history = llm_req3
        .data
        .get("history")
        .and_then(|value| value.as_array())
        .expect("llm request should include history");
    assert_eq!(history.len(), 3);
    assert_eq!(history[0]["content"], "try the command");
    assert_eq!(history[1]["content"], "The tool failed previously.");
    assert_eq!(history[2]["content"], "what happened before?");
    assert!(
        !history
            .iter()
            .any(|entry| entry["content"] == "error: permission denied")
    );

    send_text_round(&test_end, &llm_req3, "I can see the prior tool failure.").await;
    let turn2_frames = collect_response_frames(&mut test_end, turn2.id).await;
    assert!(turn2_frames.iter().any(|frame| {
        frame.status == Status::Item
            && frame.data.get("type").and_then(|value| value.as_str()) == Some("reply")
            && frame.data.get("content").and_then(|value| value.as_str())
                == Some("I can see the prior tool failure.")
    }));
    assert_eq!(
        turn2_frames.last().map(|frame| frame.status),
        Some(Status::Done)
    );

    drop(test_end);
    dispatch_task.abort();
    let _ = dispatch_task.await;
}

fn request_with_data(call: &str, value: serde_json::Value) -> Frame {
    let mut frame = Frame::request(call);
    frame.data = value
        .as_object()
        .expect("request data must be an object")
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    frame
}

async fn send_to_room(test_end: &PipeEnd, frame: Frame) {
    test_end.sender().send(frame).await.unwrap();
}

async fn recv_frame(test_end: &mut PipeEnd) -> Frame {
    timeout(Duration::from_secs(2), test_end.recv())
        .await
        .expect("timed out waiting for frame")
        .expect("pipe closed unexpectedly")
}

async fn recv_request(test_end: &mut PipeEnd, call: &str) -> Frame {
    loop {
        let frame = recv_frame(test_end).await;
        if is_broadcast(&frame) {
            continue;
        }
        if frame.status == Status::Request && frame.call == call {
            return frame;
        }
    }
}

async fn collect_response_frames(test_end: &mut PipeEnd, parent_id: uuid::Uuid) -> Vec<Frame> {
    let mut frames = Vec::new();
    loop {
        let frame = recv_frame(test_end).await;
        if is_broadcast(&frame) {
            continue;
        }
        if frame.parent_id != Some(parent_id) {
            continue;
        }

        let terminal = frame.status.is_terminal();
        frames.push(frame);
        if terminal {
            return frames;
        }
    }
}

fn is_broadcast(frame: &Frame) -> bool {
    frame.status == Status::Request && frame.call.starts_with("door:")
}

async fn send_tool_use_round(test_end: &PipeEnd, llm_req: &Frame) {
    let mut tool_delta = crate::types::Data::new();
    tool_delta.insert("type".into(), "tool_use_delta".into());
    tool_delta.insert("index".into(), 0.into());
    tool_delta.insert("id".into(), "call_1".into());
    tool_delta.insert("name".into(), "syscall".into());
    tool_delta.insert(
        "input".into(),
        r#"{"syscall":"exec:run","data":{"cmd":"false"}}"#.into(),
    );

    let mut done_delta = crate::types::Data::new();
    done_delta.insert("type".into(), "done".into());
    done_delta.insert("stop_reason".into(), "tool_use".into());
    done_delta.insert("model".into(), "fake-model".into());
    done_delta.insert("input_tokens".into(), 1.into());
    done_delta.insert("output_tokens".into(), 1.into());

    send_to_room(test_end, llm_req.item(tool_delta)).await;
    send_to_room(test_end, llm_req.item(done_delta)).await;
    send_to_room(test_end, llm_req.done()).await;
}

async fn send_text_round(test_end: &PipeEnd, llm_req: &Frame, text: &str) {
    let mut text_delta = crate::types::Data::new();
    text_delta.insert("type".into(), "text_delta".into());
    text_delta.insert("text".into(), text.into());

    let mut done_delta = crate::types::Data::new();
    done_delta.insert("type".into(), "done".into());
    done_delta.insert("stop_reason".into(), "end_turn".into());
    done_delta.insert("model".into(), "fake-model".into());
    done_delta.insert("input_tokens".into(), 1.into());
    done_delta.insert("output_tokens".into(), 1.into());

    send_to_room(test_end, llm_req.item(text_delta)).await;
    send_to_room(test_end, llm_req.item(done_delta)).await;
    send_to_room(test_end, llm_req.done()).await;
}

async fn send_error_response(test_end: &PipeEnd, request: &Frame, err: &impl ErrorCode) {
    send_to_room(test_end, request.error_from(err)).await;
}
