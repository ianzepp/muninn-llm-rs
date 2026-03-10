//! LLM profile configuration loader.
//!
//! DESIGN
//! ======
//! - `load_config(path)` parses a TOML file directly — used in tests and
//!   when the host binary provides an explicit path.
//! - `resolve_api_key(profile)` checks `api_key_value` first (literal key),
//!   then reads `api_key_env` as an env var name. Literal keys are for local
//!   overrides only and must never be committed to version control.
//!
//! TOML FORMAT
//! ===========
//! ```toml
//! active = "default"
//! traits = ["ego/senior"]
//!
//! [configs.default]
//! provider = "anthropic"
//! model = "claude-sonnet-4-6"
//! max_tokens = 8096
//! api_key_env = "ANTHROPIC_API_KEY"
//! self_prompt = "You are a helpful assistant."
//!
//! [configs.hand]
//! provider = "openai"
//! model = "gpt-4o"
//! max_tokens = 4096
//! api_key_env = "OPENAI_API_KEY"
//! self_prompt = "You are a specialized worker."
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::LlmError;

// =============================================================================
// TYPES
// =============================================================================

/// A single LLM provider profile.
#[derive(Debug, Clone, Deserialize)]
pub struct LlmProfile {
    /// "anthropic" or "openai"
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    /// Env var name holding the API key (recommended).
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    /// Literal API key value (local overrides only, not for VCS).
    #[serde(default)]
    pub api_key_value: Option<String>,
    /// OpenAI API mode. Only "chat_completions" is currently supported.
    #[serde(default)]
    pub openai_api: Option<String>,
    /// OpenAI-compatible base URL (default: <https://api.openai.com>).
    #[serde(default)]
    pub openai_base_url: Option<String>,
    /// System prompt slot 0: persistent identity.
    pub self_prompt: String,
}

fn default_api_key_env() -> String {
    "ANTHROPIC_API_KEY".to_string()
}

/// Top-level config file structure.
#[derive(Debug, Clone)]
pub struct ConfigFile {
    pub active: String,
    pub configs: HashMap<String, LlmProfile>,
    /// Personality trait names in "category/variant" format (slot 9).
    pub traits: Vec<String>,
}

// =============================================================================
// DESERIALIZATION HELPERS
// =============================================================================

/// Raw TOML shape — wraps the user-facing fields before constructing `ConfigFile`.
#[derive(Deserialize)]
struct RawConfig {
    active: String,
    configs: HashMap<String, LlmProfile>,
    #[serde(default)]
    traits: Vec<String>,
}

// =============================================================================
// LOADING
// =============================================================================

/// Load and parse a config from a TOML file at the given path.
///
/// # Errors
///
/// Returns [`LlmError::ConfigLoad`] if the file cannot be read, or
/// [`LlmError::ConfigParse`] if it cannot be parsed.
pub fn load_config(path: &Path) -> Result<ConfigFile, LlmError> {
    let content = std::fs::read_to_string(path).map_err(|e| LlmError::ConfigLoad(e.to_string()))?;
    parse_config(&content)
}

/// Parse a config from a TOML string.
///
/// # Errors
///
/// Returns [`LlmError::ConfigParse`] if the TOML is malformed or missing required fields.
pub fn parse_config(content: &str) -> Result<ConfigFile, LlmError> {
    let raw: RawConfig =
        toml::from_str(content).map_err(|e| LlmError::ConfigParse(e.to_string()))?;
    Ok(ConfigFile {
        active: raw.active,
        configs: raw.configs,
        traits: raw.traits,
    })
}

/// Resolve the API key for a profile: literal value takes precedence, then env var.
///
/// # Errors
///
/// Returns [`LlmError::MissingApiKey`] if the env var is not set and no literal
/// value is provided.
pub fn resolve_api_key(profile: &LlmProfile) -> Result<String, LlmError> {
    if let Some(value) = profile.api_key_value.as_deref() {
        if !value.trim().is_empty() {
            return Ok(value.to_string());
        }
    }
    if profile.api_key_env.trim().is_empty() {
        return Err(LlmError::MissingApiKey {
            var: "configs.<name>.api_key_env".to_string(),
        });
    }
    std::env::var(&profile.api_key_env).map_err(|_| LlmError::MissingApiKey {
        var: profile.api_key_env.clone(),
    })
}

#[cfg(test)]
#[path = "config_test.rs"]
mod tests;
