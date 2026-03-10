use super::{OpenAiStreamParser, build_cc_messages};
use crate::types::{Content, ContentBlock, ContentDelta, Message};

#[test]
fn build_cc_messages_preserves_tool_result_adjacency() {
    let messages = vec![
        Message {
            role: "assistant".to_string(),
            content: Content::Blocks(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                input: serde_json::json!({"q": "x"}),
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
    ];

    let out = build_cc_messages("", &messages);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].role, "assistant");
    assert_eq!(out[1].role, "tool");
}

#[test]
fn parser_emits_multiple_tool_calls_and_usage_done() {
    let mut parser = OpenAiStreamParser::default();
    let chunk = concat!(
        "data: {\"model\":\"gpt-test\",\"choices\":[{\"delta\":{\"tool_calls\":[",
        "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"a\\\":\"}},",
        "{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"b\\\":\"}}",
        "]},\"finish_reason\":null}]}\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
        "{\"index\":0,\"function\":{\"arguments\":\"1}\"}},",
        "{\"index\":1,\"function\":{\"arguments\":\"2}\"}}",
        "]},\"finish_reason\":\"tool_calls\"}]}\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n",
        "data: [DONE]\n"
    );

    let deltas = parser.push_chunk(chunk.as_bytes()).unwrap();
    assert_eq!(
        deltas
            .iter()
            .filter(|d| matches!(d, ContentDelta::ToolUseDelta { .. }))
            .count(),
        4
    );
    let done = deltas
        .into_iter()
        .find(|delta| matches!(delta, ContentDelta::Done { .. }))
        .or_else(|| parser.finish().unwrap())
        .expect("done delta");
    assert!(matches!(
        done,
        ContentDelta::Done {
            input_tokens: 11,
            output_tokens: 7,
            ..
        }
    ));
}
