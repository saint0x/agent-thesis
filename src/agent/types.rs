//! Types for the agent

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::analyzer::QueryAnalysis;
use crate::context::ContextEngineConfig;
use crate::llm::LlmConfig;
use crate::prompt::{ModuleConfig, ComposedPrompt};

/// Configuration for the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Configuration for the small (analysis) LLM
    pub small_llm: LlmConfig,

    /// Configuration for the large (generation) LLM
    pub large_llm: LlmConfig,

    /// Prompt module configuration
    #[serde(default)]
    pub prompt_config: ModuleConfig,

    /// Context engine configuration
    #[serde(default)]
    pub context_config: ContextEngineConfig,

    /// Embedding dimension
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Whether to enable debug output
    #[serde(default)]
    pub debug: bool,

    /// Session ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Persistence path (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistence_path: Option<String>,
}

fn default_embedding_dim() -> usize { 384 }

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            small_llm: LlmConfig {
                provider: "anthropic".to_string(),
                model: "claude-3-haiku-20240307".to_string(),
                max_tokens: 1024,
                temperature: 0.1,
                ..Default::default()
            },
            large_llm: LlmConfig {
                provider: "anthropic".to_string(),
                model: "claude-3-5-sonnet-20241022".to_string(),
                max_tokens: 4096,
                temperature: 0.7,
                ..Default::default()
            },
            prompt_config: ModuleConfig::default(),
            context_config: ContextEngineConfig::default(),
            embedding_dim: default_embedding_dim(),
            debug: false,
            session_id: None,
            persistence_path: None,
        }
    }
}

/// Response from the agent
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// The generated content
    pub content: String,

    /// Query analysis results
    pub analysis: QueryAnalysis,

    /// The composed prompt used (for debugging)
    pub composed_prompt: ComposedPrompt,

    /// Performance metrics
    pub metrics: AgentMetrics,

    /// Session state snapshot
    pub state_snapshot: StateSnapshot,
}

/// Performance metrics for a request
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// Total request time in milliseconds
    pub total_time_ms: u64,

    /// Time spent on query analysis
    pub analysis_time_ms: u64,

    /// Time spent on prompt composition
    pub composition_time_ms: u64,

    /// Time spent on context retrieval
    pub retrieval_time_ms: u64,

    /// Time spent on LLM generation
    pub generation_time_ms: u64,

    /// Input tokens used (small LLM)
    pub small_llm_input_tokens: usize,

    /// Output tokens used (small LLM)
    pub small_llm_output_tokens: usize,

    /// Input tokens used (large LLM)
    pub large_llm_input_tokens: usize,

    /// Output tokens used (large LLM)
    pub large_llm_output_tokens: usize,

    /// Number of context objects retrieved
    pub context_objects_retrieved: usize,

    /// Number of prompt modules activated
    pub modules_activated: usize,
}

impl AgentMetrics {
    pub fn total_tokens(&self) -> usize {
        self.small_llm_input_tokens
            + self.small_llm_output_tokens
            + self.large_llm_input_tokens
            + self.large_llm_output_tokens
    }
}

/// Snapshot of session state
#[derive(Debug, Clone, Default)]
pub struct StateSnapshot {
    /// Current turn number
    pub turn_number: usize,

    /// Total context objects stored
    pub context_objects_count: usize,

    /// Total tokens in context store
    pub context_tokens: usize,

    /// Session ID
    pub session_id: Option<String>,
}

/// Builder for agent configuration
pub struct AgentConfigBuilder {
    config: AgentConfig,
}

impl AgentConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
        }
    }

    pub fn small_llm(mut self, provider: &str, model: &str) -> Self {
        self.config.small_llm.provider = provider.to_string();
        self.config.small_llm.model = model.to_string();
        self
    }

    pub fn large_llm(mut self, provider: &str, model: &str) -> Self {
        self.config.large_llm.provider = provider.to_string();
        self.config.large_llm.model = model.to_string();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.config.small_llm.api_key = Some(key.to_string());
        self.config.large_llm.api_key = Some(key.to_string());
        self
    }

    pub fn base_system_prompt(mut self, prompt: &str) -> Self {
        self.config.prompt_config.base_system_prompt = prompt.to_string();
        self
    }

    pub fn context_token_budget(mut self, budget: usize) -> Self {
        self.config.context_config.token_budget = budget;
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.config.context_config.top_k = k;
        self
    }

    pub fn session_id(mut self, id: &str) -> Self {
        self.config.session_id = Some(id.to_string());
        self
    }

    pub fn persistence_path(mut self, path: &str) -> Self {
        self.config.persistence_path = Some(path.to_string());
        self
    }

    pub fn debug(mut self, enabled: bool) -> Self {
        self.config.debug = enabled;
        self
    }

    pub fn build(self) -> AgentConfig {
        self.config
    }
}

impl Default for AgentConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
