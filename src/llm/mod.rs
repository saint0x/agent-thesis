//! LLM Provider interfaces and implementations
//!
//! This module provides abstractions for interacting with both small (fast) and large (capable) LLMs.

mod provider;
mod anthropic;
mod openai;

pub use provider::{LlmProvider, LlmConfig, ModelInfo, CompletionOptions, CompletionResponse};
pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;

use crate::{DpaError, Result};

/// Model size classification for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// Small, fast models for analysis (e.g., Haiku, GPT-4o-mini)
    Small,
    /// Large, capable models for generation (e.g., Sonnet, GPT-4o)
    Large,
}

/// Create a provider based on configuration
pub async fn create_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    match config.provider.as_str() {
        "anthropic" => Ok(Box::new(AnthropicProvider::new(config.clone())?)),
        "openai" => Ok(Box::new(OpenAIProvider::new(config.clone())?)),
        other => Err(DpaError::ConfigError(format!("Unknown provider: {}", other))),
    }
}
