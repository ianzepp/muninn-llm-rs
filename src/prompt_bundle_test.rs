use super::*;

#[test]
fn empty_slots_are_omitted() {
    let ctx = PromptContext {
        config: "default",
        self_prompt: "I am an assistant.",
        tools: None,
        room: "test",
        description: "",
        notes: "",
        memory: "",
        traits: &[],
    };
    let prompt = build_system_prompt(&ctx);
    assert!(!prompt.contains("Room Description"));
    assert!(!prompt.contains("Room Notes"));
    assert!(!prompt.contains("Memory"));
    assert!(prompt.contains("I am an assistant."));
    assert!(prompt.contains("Commandments"));
}

#[test]
fn tools_slot_populated_when_tools_present() {
    let tools = vec![Tool {
        name: "vfs_read".to_string(),
        description: "Read a file".to_string(),
        input_schema: serde_json::json!({}),
    }];
    let ctx = PromptContext {
        config: "default",
        self_prompt: "Assistant.",
        tools: Some(&tools),
        room: "test",
        description: "",
        notes: "",
        memory: "",
        traits: &[],
    };
    let prompt = build_system_prompt(&ctx);
    assert!(prompt.contains("vfs_read"));
}

#[test]
fn memory_slot_populated_when_non_empty() {
    let ctx = PromptContext {
        config: "default",
        self_prompt: "Assistant.",
        tools: None,
        room: "test",
        description: "",
        notes: "",
        memory: "User prefers concise answers.",
        traits: &[],
    };
    let prompt = build_system_prompt(&ctx);
    assert!(prompt.contains("Memory"));
    assert!(prompt.contains("User prefers concise answers."));
}

#[test]
fn unknown_traits_silently_skipped() {
    let traits = vec!["unknown/trait".to_string(), "ego/senior".to_string()];
    let rendered = render_traits(&traits);
    assert!(rendered.contains("confident"));
    assert!(!rendered.contains("unknown"));
}

#[test]
fn bundle_slot_ordering() {
    let mut bundle = Bundle::default();
    bundle.set(Slot::Tone, "Z-tone");
    bundle.set(Slot::Identity, "A-identity");
    let rendered = bundle.render();
    let identity_pos = rendered.find("A-identity").unwrap();
    let tone_pos = rendered.find("Z-tone").unwrap();
    assert!(
        identity_pos < tone_pos,
        "Identity (slot 0) must precede Tone (slot 9)"
    );
}
