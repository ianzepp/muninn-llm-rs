use serde_json::json;

use super::history_to_messages;
use crate::prompt_bundle::{PromptContext, build_system_prompt};
use crate::room::state::{HistoryEntry, HistoryKind};
use crate::types::{Content, ContentBlock, Message};

#[test]
fn history_to_messages_only_emits_real_turns() {
    let history = vec![
        HistoryEntry {
            id: 1,
            ts: 100,
            from: "user".to_string(),
            content: "hello".to_string(),
            kind: HistoryKind::User,
        },
        HistoryEntry {
            id: 2,
            ts: 900,
            from: "bot".to_string(),
            content: "hi".to_string(),
            kind: HistoryKind::Assistant,
        },
    ];

    let messages = history_to_messages(&history);
    assert_eq!(messages.len(), 2);
    assert!(matches!(&messages[0].content, Content::Text(text) if text == "hello"));
    assert!(matches!(&messages[1].content, Content::Text(text) if text == "hi"));
}

#[test]
fn history_to_messages_keeps_tool_followup_adjacent() {
    let history = vec![HistoryEntry {
        id: 1,
        ts: 100,
        from: "user".to_string(),
        content: "solve this".to_string(),
        kind: HistoryKind::User,
    }];

    let mut messages = history_to_messages(&history);
    messages.extend([
        Message {
            role: "assistant".to_string(),
            content: Content::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                input: json!({"q": "x"}),
            }]),
        },
        Message {
            role: "user".to_string(),
            content: Content::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "ok".to_string(),
                is_error: Some(false),
            }]),
        },
    ]);

    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[2].role, "user");
    assert!(
        matches!(&messages[1].content, Content::Blocks(blocks) if matches!(&blocks[0], ContentBlock::ToolUse { .. }))
    );
    assert!(
        matches!(&messages[2].content, Content::Blocks(blocks) if matches!(&blocks[0], ContentBlock::ToolResult { .. }))
    );
}

#[test]
fn system_prompt_includes_request_memory_when_present() {
    let prompt = build_system_prompt(&PromptContext {
        config: "default",
        self_prompt: "Assistant.",
        tools: None,
        room: "general",
        description: "",
        notes: "",
        memory: "Prior tool outcome: exec:run failed with permission denied.",
        traits: &[],
    });

    assert!(prompt.contains("Memory"));
    assert!(prompt.contains("permission denied"));
}
