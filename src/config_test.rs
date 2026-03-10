use super::*;

#[test]
fn parse_minimal_config() {
    let toml = r#"
active = "default"

[configs.default]
provider = "anthropic"
model = "claude-sonnet-4-6"
max_tokens = 8096
api_key_env = "ANTHROPIC_API_KEY"
self_prompt = "You are a helpful assistant."
"#;
    let cfg = parse_config(toml).unwrap();
    assert_eq!(cfg.active, "default");
    assert!(cfg.configs.contains_key("default"));
    assert!(cfg.traits.is_empty());
}

#[test]
fn parse_config_with_traits() {
    let toml = r#"
active = "default"
traits = ["ego/senior", "filter/slack"]

[configs.default]
provider = "anthropic"
model = "claude-sonnet-4-6"
max_tokens = 4096
api_key_env = "ANTHROPIC_API_KEY"
self_prompt = "Assistant."
"#;
    let cfg = parse_config(toml).unwrap();
    assert_eq!(cfg.traits, vec!["ego/senior", "filter/slack"]);
}

#[test]
fn resolve_literal_key() {
    let profile = LlmProfile {
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4-6".to_string(),
        max_tokens: 4096,
        api_key_env: String::new(),
        api_key_value: Some("sk-literal".to_string()),
        openai_api: None,
        openai_base_url: None,
        self_prompt: String::new(),
    };
    assert_eq!(resolve_api_key(&profile).unwrap(), "sk-literal");
}

#[test]
fn resolve_missing_key_errors() {
    let profile = LlmProfile {
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4-6".to_string(),
        max_tokens: 4096,
        api_key_env: "MUNINN_NONEXISTENT_KEY_12345".to_string(),
        api_key_value: None,
        openai_api: None,
        openai_base_url: None,
        self_prompt: String::new(),
    };
    assert!(resolve_api_key(&profile).is_err());
}

#[test]
fn parse_openai_api_mode() {
    let toml = r#"
active = "default"

[configs.default]
provider = "openai"
model = "gpt-4o"
max_tokens = 4096
api_key_env = "OPENAI_API_KEY"
openai_api = "responses"
self_prompt = "Assistant."
"#;
    let cfg = parse_config(toml).unwrap();
    assert_eq!(
        cfg.configs["default"].openai_api.as_deref(),
        Some("responses")
    );
}
