//! Embedding provider implementations

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::llm::LlmProvider;
use crate::Result;

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for the given texts
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;
}

/// LLM-backed embedding provider
pub struct LlmEmbeddingProvider {
    llm: Arc<dyn LlmProvider>,
    dimension: usize,
}

impl LlmEmbeddingProvider {
    pub fn new(llm: Arc<dyn LlmProvider>, dimension: usize) -> Self {
        Self { llm, dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for LlmEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.llm.embed(texts).await
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Simple hash-based embedding provider for testing/development
/// This creates deterministic pseudo-embeddings based on text content
pub struct SimpleEmbeddingProvider {
    dimension: usize,
}

impl SimpleEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic pseudo-embedding from text
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = Vec::with_capacity(self.dimension);

        // Use multiple hash seeds to create different dimensions
        for seed in 0..self.dimension {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            text.hash(&mut hasher);
            let hash = hasher.finish();

            // Convert hash to float in [-1, 1] range
            let value = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            embedding.push(value);
        }

        // Normalize the vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut embedding {
                *v /= magnitude;
            }
        }

        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for SimpleEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_embed(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_embedding_determinism() {
        let provider = SimpleEmbeddingProvider::new(128);

        let texts = vec!["hello world".to_string()];
        let emb1 = provider.embed(&texts).await.unwrap();
        let emb2 = provider.embed(&texts).await.unwrap();

        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_simple_embedding_dimension() {
        let provider = SimpleEmbeddingProvider::new(256);

        let texts = vec!["test".to_string()];
        let embeddings = provider.embed(&texts).await.unwrap();

        assert_eq!(embeddings[0].len(), 256);
    }
}
