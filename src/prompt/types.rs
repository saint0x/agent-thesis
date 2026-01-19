//! Types for the dynamic prompt builder

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A prompt module that can be conditionally included
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptModule {
    /// Unique identifier for the module
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// The prompt content (can include Handlebars templates)
    pub content: String,

    /// Description of what this module does
    #[serde(default)]
    pub description: String,

    /// Priority for ordering (higher = placed earlier in prompt)
    #[serde(default = "default_priority")]
    pub priority: i32,

    /// Trigger conditions for activation
    #[serde(default)]
    pub triggers: Vec<super::triggers::Trigger>,

    /// IDs of modules that cannot be used together with this one
    #[serde(default)]
    pub mutual_exclusions: Vec<String>,

    /// IDs of modules that must also be included if this one is
    #[serde(default)]
    pub dependencies: Vec<String>,

    /// Whether this module is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Section where this module should be placed
    #[serde(default)]
    pub section: PromptSection,

    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_priority() -> i32 { 0 }
fn default_enabled() -> bool { true }

impl Default for PromptModule {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            content: String::new(),
            description: String::new(),
            priority: 0,
            triggers: Vec::new(),
            mutual_exclusions: Vec::new(),
            dependencies: Vec::new(),
            enabled: true,
            section: PromptSection::System,
            metadata: HashMap::new(),
        }
    }
}

/// Section of the prompt where a module can be placed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptSection {
    /// System/instruction section
    #[default]
    System,

    /// Context section (before user query)
    Context,

    /// Examples section (few-shot)
    Examples,

    /// User query section
    Query,

    /// Suffix section (after query)
    Suffix,
}

/// A composed prompt ready to send to the LLM
#[derive(Debug, Clone, Default)]
pub struct ComposedPrompt {
    /// System prompt content
    pub system: String,

    /// Context content (conversation history, retrieved context)
    pub context: String,

    /// Few-shot examples
    pub examples: Vec<Example>,

    /// User query
    pub query: String,

    /// Suffix content
    pub suffix: String,

    /// Total estimated token count
    pub estimated_tokens: usize,

    /// IDs of activated modules
    pub activated_modules: Vec<String>,

    /// Metadata about composition
    pub composition_metadata: CompositionMetadata,
}

impl ComposedPrompt {
    /// Create a new empty composed prompt
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the full prompt as a single string (for models that don't separate system)
    pub fn as_single_prompt(&self) -> String {
        let mut parts = Vec::new();

        if !self.system.is_empty() {
            parts.push(self.system.clone());
        }

        if !self.context.is_empty() {
            parts.push(self.context.clone());
        }

        for example in &self.examples {
            parts.push(format!("User: {}\nAssistant: {}", example.user, example.assistant));
        }

        if !self.query.is_empty() {
            parts.push(format!("User: {}", self.query));
        }

        if !self.suffix.is_empty() {
            parts.push(self.suffix.clone());
        }

        parts.join("\n\n")
    }

    /// Get the user message portion (context + examples + query)
    pub fn user_message(&self) -> String {
        let mut parts = Vec::new();

        if !self.context.is_empty() {
            parts.push(format!("<context>\n{}\n</context>", self.context));
        }

        for example in &self.examples {
            parts.push(format!(
                "<example>\nUser: {}\nAssistant: {}\n</example>",
                example.user, example.assistant
            ));
        }

        parts.push(self.query.clone());

        if !self.suffix.is_empty() {
            parts.push(self.suffix.clone());
        }

        parts.join("\n\n")
    }
}

/// A few-shot example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// User input
    pub user: String,

    /// Assistant response
    pub assistant: String,

    /// Topics this example is relevant for
    #[serde(default)]
    pub topics: Vec<String>,

    /// Embedding for similarity matching (optional)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

/// Metadata about how the prompt was composed
#[derive(Debug, Clone, Default)]
pub struct CompositionMetadata {
    /// Time taken to compose in milliseconds
    pub composition_time_ms: u64,

    /// Number of modules evaluated
    pub modules_evaluated: usize,

    /// Number of modules activated
    pub modules_activated: usize,

    /// Number of context objects included
    pub context_objects_included: usize,

    /// Number of examples included
    pub examples_included: usize,

    /// Token budget used
    pub tokens_used: usize,

    /// Token budget available
    pub token_budget: usize,
}

/// Configuration for loading modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfig {
    /// Directory containing module TOML files
    #[serde(default = "default_modules_dir")]
    pub modules_dir: String,

    /// Base system prompt (always included)
    #[serde(default)]
    pub base_system_prompt: String,

    /// Default modules to always include
    #[serde(default)]
    pub default_modules: Vec<String>,

    /// Token budget for prompt modules
    #[serde(default = "default_module_token_budget")]
    pub module_token_budget: usize,

    /// Token budget for context
    #[serde(default = "default_context_token_budget")]
    pub context_token_budget: usize,

    /// Token budget for examples
    #[serde(default = "default_example_token_budget")]
    pub example_token_budget: usize,

    /// Maximum number of examples to include
    #[serde(default = "default_max_examples")]
    pub max_examples: usize,
}

fn default_modules_dir() -> String { "modules".to_string() }
fn default_module_token_budget() -> usize { 2000 }
fn default_context_token_budget() -> usize { 8000 }
fn default_example_token_budget() -> usize { 2000 }
fn default_max_examples() -> usize { 3 }

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            modules_dir: default_modules_dir(),
            base_system_prompt: String::new(),
            default_modules: Vec::new(),
            module_token_budget: default_module_token_budget(),
            context_token_budget: default_context_token_budget(),
            example_token_budget: default_example_token_budget(),
            max_examples: default_max_examples(),
        }
    }
}
