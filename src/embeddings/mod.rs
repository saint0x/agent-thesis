//! Embedding system for vector similarity search
//!
//! This module provides:
//! - Embedding generation via LLM providers or local models
//! - Vector store with HNSW indexing for fast similarity search
//! - Similarity scoring functions

mod provider;
mod store;
mod similarity;

pub use provider::EmbeddingProvider;
pub use store::{VectorStore, VectorEntry};
pub use similarity::{cosine_similarity, normalize_vector};

use crate::Result;
use std::sync::Arc;

/// Create a default embedding provider
pub fn create_embedding_provider(
    dimension: usize,
) -> Result<Arc<dyn EmbeddingProvider>> {
    Ok(Arc::new(provider::SimpleEmbeddingProvider::new(dimension)))
}
