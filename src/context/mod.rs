//! Modular Context Engine
//!
//! This module provides:
//! - Context Object (CO) definitions and storage
//! - Lossless context partitioning using small LLM
//! - Relevance scoring for context retrieval
//! - Context composition for prompt building

mod types;
mod store;
mod partitioner;
mod scorer;
mod engine;

pub use types::{ContextObject, ContextType, ContextMetadata};
pub use store::ContextStore;
pub use partitioner::ContextPartitioner;
pub use scorer::{RelevanceScorer, ScoringWeights};
pub use engine::{ContextEngine, ContextEngineConfig, ContextSummary};
