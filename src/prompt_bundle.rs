//! Layered system prompt assembly for LLM requests.
//!
//! WHY 10 semantic slots: different concerns (identity, constraints, context,
//! tools, room description, notes, environment, memory, tone) change
//! independently and at different rates. Each slot has a fixed purpose.
//! The final prompt is all non-empty slots joined with `\n\n` in slot order.
//!
//! WHY fixed slots over a dynamic vec: slots have defined semantics and ordering.
//! A dynamic vec would lose the guarantee that identity always precedes tools,
//! or that tone is always last. The enum makes the contract compile-time checked.
//!
//! WHY no per-request data in the system prompt: the system prompt must stay
//! identical across requests to benefit from provider-side prompt caching.

use crate::types::Tool;

// =============================================================================
// SLOT ENUM
// =============================================================================

/// Semantic prompt slots ordered from most stable to most behavioral.
///
/// Gaps in numbering reserve positions for future layers without shuffling
/// existing slot positions (which would invalidate cached prompts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Slot {
    /// Persistent identity — from config `self_prompt`.
    Identity = 0,
    /// Shared invariants — safety constraints baked in at compile time.
    Commandments = 1,
    /// Role-specific context — behavioral constraints derived from config name.
    Context = 2,
    /// Primary tools — syscall reference and available tool names.
    ToolsPrimary = 3,
    /// Room description — purpose from caller-provided description text.
    Description = 4,
    /// Room notes — accumulated context from caller-provided notes text.
    Notes = 5,
    // Slot 6 reserved for future behavioral constraints.
    /// Environment — OS and room name.
    Environment = 7,
    /// Long-term memory — left empty by this crate; host may pre-fill.
    Memory = 8,
    /// Personality traits — composable tone modifiers.
    Tone = 9,
}

const SLOT_COUNT: usize = 10;

// =============================================================================
// BUNDLE
// =============================================================================

/// A fixed-size array of optional prompt sections rendered in slot order.
#[derive(Debug, Default)]
pub struct Bundle {
    slots: [Option<String>; SLOT_COUNT],
}

impl Bundle {
    /// Set a slot's content. Empty or whitespace-only content clears the slot.
    pub fn set(&mut self, slot: Slot, content: impl Into<String>) {
        let text = content.into();
        let trimmed = text.trim();
        self.slots[slot as usize] = if trimmed.is_empty() { None } else { Some(trimmed.to_string()) };
    }

    /// Join all non-empty slots with double newlines.
    #[must_use]
    pub fn render(self) -> String {
        self.slots.into_iter().flatten().collect::<Vec<_>>().join("\n\n")
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Context required to assemble a system prompt.
pub struct PromptContext<'a> {
    /// Profile/config name (e.g., "default", "hand") — used for slot 2 context.
    pub config: &'a str,
    /// Persistent identity text (slot 0).
    pub self_prompt: &'a str,
    /// Tool definitions for slot 3 (optional).
    pub tools: Option<&'a [Tool]>,
    /// Room name for slot 7 environment block.
    pub room: &'a str,
    /// Room description text for slot 4 (empty = slot omitted).
    pub description: &'a str,
    /// Room notes text for slot 5 (empty = slot omitted).
    pub notes: &'a str,
    /// Long-term memory text for slot 8 (empty = slot omitted).
    /// This crate never populates this; the host may pre-fill.
    pub memory: &'a str,
    /// Personality trait names in "category/variant" format for slot 9.
    pub traits: &'a [String],
}

/// Assemble a cacheable system prompt from stable context.
///
/// The result changes only when the context fields change — not on every
/// request — so it is safe to use as a provider cache key.
#[must_use]
pub fn build_system_prompt(ctx: &PromptContext<'_>) -> String {
    let mut bundle = Bundle::default();

    bundle.set(Slot::Identity, ctx.self_prompt);
    bundle.set(Slot::Commandments, include_str!("bundler/commandments.md"));
    bundle.set(Slot::Context, render_context_slot(ctx.config));
    bundle.set(Slot::ToolsPrimary, render_tools_slot(ctx.tools));
    bundle.set(Slot::Description, render_description_slot(ctx.description));
    bundle.set(Slot::Notes, render_notes_slot(ctx.notes));
    bundle.set(Slot::Environment, render_environment_slot(ctx.room));
    bundle.set(Slot::Memory, render_memory_slot(ctx.memory));
    bundle.set(Slot::Tone, render_traits(ctx.traits));

    bundle.render()
}

// =============================================================================
// SLOT RENDERERS
// =============================================================================

/// Role-specific context text selected by config name.
///
/// WHY config name as role key: the config name already determines the model
/// and parameters. Tying the role context to the same name means no extra
/// field is needed. Unknown names default to the generic assistant context.
fn render_context_slot(config: &str) -> &'static str {
    match config {
        "hand" => "## Role: Hand\n\nYou are a specialized worker actor. \
                   You execute specific tasks delegated by a head actor. \
                   Be precise, efficient, and return structured results.",
        "mind" => "## Role: Mind\n\nYou are a research and reasoning actor. \
                   You analyze problems deeply and produce detailed reports \
                   for head actors to act on.",
        "work" => "## Role: Work\n\nYou are a background processing actor. \
                   You handle long-running or batch tasks autonomously.",
        _ => "## Role: Assistant\n\nYou are a helpful assistant. \
              You respond to user messages, use tools when helpful, \
              and delegate to specialist actors when appropriate.",
    }
}

fn render_tools_slot(tools: Option<&[Tool]>) -> String {
    let Some(specs) = tools else {
        return String::new();
    };
    if specs.is_empty() {
        return String::new();
    }
    let names: Vec<&str> = specs.iter().map(|t| t.name.as_str()).collect();
    format!(
        "## Tools\n\nYou have access to the following tools: {}.\n\n\
         Call tools by name with a JSON input matching their schema. \
         Prefer targeted tool calls over broad ones. \
         Check tool results before proceeding.",
        names.join(", ")
    )
}

fn render_description_slot(description: &str) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    format!("## Room Description\n\n{trimmed}")
}

fn render_notes_slot(notes: &str) -> String {
    let trimmed = notes.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    format!("## Room Notes\n\n{trimmed}")
}

fn render_environment_slot(room: &str) -> String {
    let mut lines = vec![
        "## Environment".to_string(),
        String::new(),
        format!("- OS: {} ({})", std::env::consts::OS, std::env::consts::ARCH),
        format!("- Room: {room}"),
        String::new(),
    ];
    // Remove trailing empty string if present
    if lines.last().map_or(false, |s: &String| s.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

fn render_memory_slot(memory: &str) -> String {
    let trimmed = memory.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    format!("## Memory\n\n{trimmed}")
}

/// Render a list of trait names into combined markdown.
///
/// Unknown trait names are silently skipped — the caller is not required to
/// validate trait names against a catalog. Missing traits simply contribute nothing.
fn render_traits(names: &[String]) -> String {
    names
        .iter()
        .filter_map(|n| lookup_trait(n))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Look up a built-in trait snippet by "category/variant" key.
///
/// Returns `None` for unknown names so callers never fail on missing traits.
fn lookup_trait(name: &str) -> Option<&'static str> {
    match name {
        "ego/intern" => Some("Be humble. Assume you may be wrong. Ask clarifying questions."),
        "ego/senior" => Some("Be confident and direct. Offer opinions when useful."),
        "ego/10x" => Some("Be extremely confident. Volunteer strong opinions freely."),
        "filter/hr" => Some("Keep all responses professional and workplace-appropriate."),
        "filter/slack" => Some("Use a casual, friendly tone suitable for team chat."),
        "filter/anon" => Some("Speak without a professional filter. Be blunt and direct."),
        "fever/mild" => Some("Be slightly more energetic and enthusiastic than baseline."),
        "fever/hot" => Some("Be very energetic, fast-paced, and enthusiastic."),
        "collab/normie" => Some("Collaborate openly. Ask for input and consensus."),
        "collab/feral" => Some("Work autonomously. Ask for clarification only when truly blocked."),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
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
        assert!(identity_pos < tone_pos, "Identity (slot 0) must precede Tone (slot 9)");
    }
}
