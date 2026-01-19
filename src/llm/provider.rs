//! LLM Provider trait and common types

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::Result;

/// Configuration for an LLM provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider name (anthropic, openai)
    pub provider: String,

    /// Model identifier
    pub model: String,

    /// API key (or use environment variable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// API base URL override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for sampling
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_max_tokens() -> usize { 4096 }
fn default_temperature() -> f32 { 0.7 }
fn default_timeout() -> u64 { 60 }

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            model: "claude-3-haiku-20240307".to_string(),
            api_key: None,
            base_url: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            timeout_secs: default_timeout(),
        }
    }
}

/// Information about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub provider: String,
    pub model: String,
    pub context_window: usize,
    pub supports_json_mode: bool,
    pub supports_streaming: bool,
}

/// Options for a completion request
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    /// System prompt
    pub system: Option<String>,

    /// Maximum tokens to generate (overrides config)
    pub max_tokens: Option<usize>,

    /// Temperature (overrides config)
    pub temperature: Option<f32>,

    /// Request JSON output
    pub json_mode: bool,

    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

/// Response from a completion request
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated content
    pub content: String,

    /// Number of input tokens
    pub input_tokens: usize,

    /// Number of output tokens
    pub output_tokens: usize,

    /// Model used
    pub model: String,

    /// Stop reason
    pub stop_reason: Option<String>,

    /// Latency in milliseconds
    pub latency_ms: u64,
}

/// Trait for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync + Debug {
    /// Generate a completion for the given prompt
    async fn complete(&self, prompt: &str, options: &CompletionOptions) -> Result<CompletionResponse>;

    /// Generate embeddings for text (if supported)
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Check if this provider supports embeddings
    fn supports_embeddings(&self) -> bool;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Get the provider name
    fn provider_name(&self) -> &str;
}
