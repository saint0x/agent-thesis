//! Dynamic Prompt Agent
//!
//! A runtime-adaptive prompt composition and modular context management agent.
//!
//! This library provides:
//! - **Query Analysis**: Small LLM-based analysis of user queries for intent, sentiment, and triggers
//! - **Dynamic Prompt Building**: Conditional prompt module composition based on query analysis
//! - **Modular Context Engine**: Lossless partitioning of context into retrievable objects
//! - **Agent Orchestration**: Full pipeline from query to response with context management

pub mod analyzer;
pub mod prompt;
pub mod context;
pub mod llm;
pub mod embeddings;
pub mod agent;

// Re-export main types for convenience
pub use agent::{Agent, AgentConfig, AgentResponse, AgentMetrics, ConversationState};
pub use analyzer::{QueryAnalyzer, QueryAnalysis, Intent, Sentiment};
pub use prompt::{PromptBuilder, PromptModule, Trigger, ComposedPrompt, ModuleConfig, create_default_modules};
pub use context::{ContextEngine, ContextObject, ContextType, ContextStore, RelevanceScorer, ScoringWeights};
pub use llm::{LlmProvider, LlmConfig, ModelInfo, create_provider, AnthropicProvider, OpenAIProvider};
pub use embeddings::{EmbeddingProvider, VectorStore, VectorEntry, cosine_similarity};

use thiserror::Error;

/// Main error type for the Dynamic Prompt Agent
#[derive(Error, Debug)]
pub enum DpaError {
    #[error("LLM provider error: {0}")]
    LlmError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Context store error: {0}")]
    ContextError(String),

    #[error("Prompt composition error: {0}")]
    PromptError(String),

    #[error("Query analysis error: {0}")]
    AnalysisError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Template error: {0}")]
    TemplateError(#[from] handlebars::RenderError),
}

pub type Result<T> = std::result::Result<T, DpaError>;

/// Global configuration constants
pub mod constants {
    /// Default token budget for context
    pub const DEFAULT_CONTEXT_TOKEN_BUDGET: usize = 8000;

    /// Default token budget for prompt modules
    pub const DEFAULT_PROMPT_TOKEN_BUDGET: usize = 2000;

    /// Maximum tokens per context object
    pub const MAX_CONTEXT_OBJECT_TOKENS: usize = 1000;

    /// Minimum tokens per context object
    pub const MIN_CONTEXT_OBJECT_TOKENS: usize = 50;

    /// Default embedding dimension
    pub const DEFAULT_EMBEDDING_DIM: usize = 384;

    /// Default number of context objects to retrieve
    pub const DEFAULT_TOP_K: usize = 5;

    /// Recency decay half-life in hours
    pub const RECENCY_HALF_LIFE_HOURS: f64 = 24.0;
}
