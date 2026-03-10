use super::{AnthropicStreamParser, reconstruct_content_blocks};
use crate::types::{ContentBlock, ContentDelta};

#[test]
fn parser_keeps_tool_fragments_separate_by_index() {
    let mut parser = AnthropicStreamParser::default();
    let chunk = concat!(
        "event: content_block_start\n",
        "data: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"lookup\"}}\n\n",
        "event: content_block_start\n",
        "data: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_2\",\"name\":\"search\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"b\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"1}\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"2}\"}}\n\n"
    );

    let deltas = parser.push_chunk(chunk.as_bytes()).unwrap();
    let (blocks, _) = reconstruct_content_blocks(&deltas).unwrap();
    assert!(matches!(&blocks[0], ContentBlock::ToolUse { id, .. } if id == "tool_1"));
    assert!(matches!(&blocks[1], ContentBlock::ToolUse { id, .. } if id == "tool_2"));
}

#[test]
fn reconstruct_content_blocks_handles_multiple_tool_indices() {
    let deltas = vec![
        ContentDelta::ToolUseDelta {
            index: 0,
            id: "tool_1".to_string(),
            name: "lookup".to_string(),
            input_fragment: "{\"a\":".to_string(),
        },
        ContentDelta::ToolUseDelta {
            index: 1,
            id: "tool_2".to_string(),
            name: "search".to_string(),
            input_fragment: "{\"b\":".to_string(),
        },
        ContentDelta::ToolUseDelta {
            index: 0,
            id: "tool_1".to_string(),
            name: "lookup".to_string(),
            input_fragment: "1}".to_string(),
        },
        ContentDelta::ToolUseDelta {
            index: 1,
            id: "tool_2".to_string(),
            name: "search".to_string(),
            input_fragment: "2}".to_string(),
        },
    ];

    let (blocks, _) = reconstruct_content_blocks(&deltas).unwrap();
    assert_eq!(blocks.len(), 2);
    assert!(matches!(&blocks[0], ContentBlock::ToolUse { id, .. } if id == "tool_1"));
    assert!(matches!(&blocks[1], ContentBlock::ToolUse { id, .. } if id == "tool_2"));
}

#[test]
fn reconstruct_content_blocks_rejects_invalid_tool_json() {
    let deltas = vec![ContentDelta::ToolUseDelta {
        index: 0,
        id: "tool_1".to_string(),
        name: "lookup".to_string(),
        input_fragment: "{\"a\":".to_string(),
    }];

    let err = reconstruct_content_blocks(&deltas).unwrap_err();
    assert!(matches!(err, crate::error::LlmError::ApiParse(_)));
}
