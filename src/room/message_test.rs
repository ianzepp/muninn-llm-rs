use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::timeout;

use muninn_kernel::pipe::pipe;
use muninn_kernel::sender::FrameSender;

use super::*;

async fn recv_frame(pipe: &mut muninn_kernel::PipeEnd) -> Frame {
    timeout(Duration::from_secs(2), pipe.recv())
        .await
        .expect("timed out waiting for frame")
        .expect("pipe closed unexpectedly")
}

#[tokio::test]
async fn emit_broadcasts_sends_thought_and_tool_frames() {
    let (mut tx_end, mut rx_end) = pipe(8);
    let caller = tx_end.caller();
    let blocks = vec![
        ContentBlock::Thinking {
            thinking: "Need a tool".to_string(),
        },
        ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "exec".to_string(),
            input: serde_json::json!({"syscall":"exec:run","data":{"cmd":"pwd"}}),
        },
    ];

    emit_broadcasts(&caller, "general", "bot-1", &blocks).await;

    let thought = recv_frame(&mut rx_end).await;
    assert_eq!(thought.call, "door:thought");
    assert_eq!(thought.from.as_deref(), Some("bot-1"));
    assert_eq!(
        thought.data.get("room").and_then(|v| v.as_str()),
        Some("general")
    );
    assert_eq!(
        thought.data.get("content").and_then(|v| v.as_str()),
        Some("Need a tool")
    );

    let tool = recv_frame(&mut rx_end).await;
    assert_eq!(tool.call, "door:tool");
    assert_eq!(tool.from.as_deref(), Some("bot-1"));
    assert_eq!(
        tool.data.get("room").and_then(|v| v.as_str()),
        Some("general")
    );
    assert_eq!(
        tool.data.get("syscall").and_then(|v| v.as_str()),
        Some("exec:run")
    );
    assert_eq!(
        tool.data
            .get("args")
            .and_then(|v| v.get("cmd"))
            .and_then(|v| v.as_str()),
        Some("pwd")
    );
}

#[tokio::test]
async fn emit_chat_sends_chat_frame() {
    let (mut tx_end, mut rx_end) = pipe(8);
    let caller = tx_end.caller();

    emit_chat(&caller, "general", "bot-1", "done").await;

    let chat = recv_frame(&mut rx_end).await;
    assert_eq!(chat.call, "door:chat");
    assert_eq!(chat.from.as_deref(), Some("bot-1"));
    assert_eq!(
        chat.data.get("room").and_then(|v| v.as_str()),
        Some("general")
    );
    assert_eq!(
        chat.data.get("content").and_then(|v| v.as_str()),
        Some("done")
    );
}

#[tokio::test]
async fn collect_chat_deltas_propagates_error_frames() {
    let (mut caller_end, responder_end) = pipe(8);
    let caller = caller_end.caller();
    let req = Frame::request("llm:chat");
    let mut stream = caller.call(req.clone()).await.unwrap();
    let sender = responder_end.sender();

    sender.send(req.error("llm exploded")).await.unwrap();

    let (tx, _rx) = mpsc::channel(4);
    let upstream = FrameSender::new(tx);
    let err = collect_chat_deltas(&mut stream, &upstream, &req, "general", "bot-1")
        .await
        .unwrap_err();
    assert!(matches!(err, LlmError::InternalCall(_)));
    assert!(err.to_string().contains("llm exploded"));
}

#[test]
fn build_chat_frame_includes_memory_when_present() {
    let frame = build_chat_frame(
        "default",
        &[],
        &[],
        "Recent tool outcomes:\n- turn 3 | actor bot | exec:run | error | permission denied",
        &[],
        "general",
    )
    .unwrap();

    assert_eq!(
        frame.data.get("memory").and_then(|v| v.as_str()),
        Some("Recent tool outcomes:\n- turn 3 | actor bot | exec:run | error | permission denied")
    );
}
