//! Context Engine - Main interface for modular context management

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::partitioner::ContextPartitioner;
use super::scorer::{RelevanceScorer, ScoringWeights};
use super::store::ContextStore;
use super::types::{ContextObject, ContextType, PartitionResult};
use crate::analyzer::QueryAnalysis;
use crate::embeddings::EmbeddingProvider;
use crate::llm::LlmProvider;
use crate::{DpaError, Result};

/// Configuration for the context engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEngineConfig {
    /// Maximum number of context objects to retrieve
    pub top_k: usize,

    /// Token budget for retrieved context
    pub token_budget: usize,

    /// Scoring weights for retrieval
    pub scoring_weights: ScoringWeights,

    /// Recency decay half-life in hours
    pub recency_half_life_hours: f64,

    /// Whether to automatically partition incoming context
    pub auto_partition: bool,

    /// Maximum tokens per context object during partitioning
    pub max_object_tokens: usize,

    /// Minimum tokens per context object during partitioning
    pub min_object_tokens: usize,
}

impl Default for ContextEngineConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            token_budget: 8000,
            scoring_weights: ScoringWeights::default(),
            recency_half_life_hours: 24.0,
            auto_partition: true,
            max_object_tokens: 1000,
            min_object_tokens: 20,
        }
    }
}

/// Main context engine for modular context management
pub struct ContextEngine {
    /// Context object store
    store: ContextStore,

    /// Context partitioner
    partitioner: ContextPartitioner,

    /// Relevance scorer
    scorer: RelevanceScorer,

    /// Embedding provider
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Configuration
    config: ContextEngineConfig,

    /// Current session ID
    session_id: Option<String>,

    /// Current turn number
    turn_number: usize,
}

impl ContextEngine {
    /// Create a new context engine
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        config: ContextEngineConfig,
    ) -> Self {
        let scorer = RelevanceScorer::new()
            .with_weights(config.scoring_weights.clone())
            .with_recency_half_life(config.recency_half_life_hours);

        let partitioner = ContextPartitioner::new(llm)
            .with_max_object_tokens(config.max_object_tokens)
            .with_min_object_tokens(config.min_object_tokens);

        Self {
            store: ContextStore::new(embedding_provider.clone()),
            partitioner,
            scorer,
            embedding_provider,
            config,
            session_id: None,
            turn_number: 0,
        }
    }

    /// Create with persistent storage
    pub fn with_persistence<P: AsRef<std::path::Path>>(
        llm: Arc<dyn LlmProvider>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        config: ContextEngineConfig,
        db_path: P,
    ) -> Result<Self> {
        let scorer = RelevanceScorer::new()
            .with_weights(config.scoring_weights.clone())
            .with_recency_half_life(config.recency_half_life_hours);

        let partitioner = ContextPartitioner::new(llm)
            .with_max_object_tokens(config.max_object_tokens)
            .with_min_object_tokens(config.min_object_tokens);

        Ok(Self {
            store: ContextStore::with_persistence(embedding_provider.clone(), db_path)?,
            partitioner,
            scorer,
            embedding_provider,
            config,
            session_id: None,
            turn_number: 0,
        })
    }

    /// Set the session ID
    pub fn set_session(&mut self, session_id: &str) {
        self.session_id = Some(session_id.to_string());
    }

    /// Increment the turn number
    pub fn next_turn(&mut self) {
        self.turn_number += 1;
    }

    /// Get the current turn number
    pub fn turn_number(&self) -> usize {
        self.turn_number
    }

    /// Ingest raw context, partitioning it into context objects
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn ingest(&self, content: &str, source: &str) -> Result<Vec<uuid::Uuid>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let result = if self.config.auto_partition {
            self.partitioner.partition(content, source).await?
        } else {
            // Create single object without partitioning
            let obj = ContextObject::new(content.to_string(), ContextType::Other)
                .with_importance(0.5);
            PartitionResult {
                objects: vec![obj],
                is_lossless: true,
                validation: Default::default(),
            }
        };

        info!(
            "Ingested context: {} objects, lossless={}",
            result.objects.len(),
            result.is_lossless
        );

        // Insert objects into store
        self.store.insert_batch(result.objects).await
    }

    /// Ingest a user message and assistant response
    pub async fn ingest_turn(
        &self,
        user_message: &str,
        assistant_response: &str,
    ) -> Result<Vec<uuid::Uuid>> {
        let combined = format!(
            "User: {}\n\nAssistant: {}",
            user_message, assistant_response
        );

        self.ingest(&combined, "conversation").await
    }

    /// Add a pre-built context object directly
    pub async fn add_object(&self, object: ContextObject) -> Result<uuid::Uuid> {
        self.store.insert(object).await
    }

    /// Retrieve relevant context objects for a query
    #[instrument(skip(self, analysis), fields(query_len = analysis.original_query.len()))]
    pub async fn retrieve(&self, analysis: &QueryAnalysis) -> Result<Vec<ContextObject>> {
        // Generate query embedding
        let query_embeddings = self
            .embedding_provider
            .embed(&[analysis.original_query.clone()])
            .await?;
        let query_embedding = query_embeddings.into_iter().next();

        // Get query topics
        let query_topics: Vec<String> = analysis.topics.clone();

        // Search by embedding similarity first
        let mut candidates = if let Some(ref emb) = query_embedding {
            self.store
                .search_similar_objects(emb, self.config.top_k * 3)?
        } else {
            // Fallback: get all objects and score them
            let all_objects = self.store.get_all()?;
            all_objects.into_iter().map(|o| (o, 0.5)).collect()
        };

        // Re-rank using full scoring
        let mut scored: Vec<(ContextObject, f32)> = candidates
            .into_iter()
            .map(|(obj, _)| {
                let score = self.scorer.score(
                    &obj,
                    query_embedding.as_deref(),
                    &query_topics,
                    chrono::Utc::now(),
                );
                (obj, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-k within token budget
        let mut selected = Vec::new();
        let mut total_tokens = 0;

        for (mut obj, score) in scored {
            if selected.len() >= self.config.top_k {
                break;
            }

            let obj_tokens = obj.token_count;
            if total_tokens + obj_tokens > self.config.token_budget {
                // Try to fit by using summary if available
                if obj.summary.is_some() {
                    obj.is_compressed = true;
                    obj.token_count = obj.summary.as_ref().map(|s| s.len() / 4).unwrap_or(0);

                    if total_tokens + obj.token_count <= self.config.token_budget {
                        total_tokens += obj.token_count;
                        obj.mark_accessed();
                        selected.push(obj);
                    }
                }
                continue;
            }

            total_tokens += obj_tokens;
            obj.mark_accessed();
            selected.push(obj);
        }

        debug!(
            "Retrieved {} context objects ({} tokens)",
            selected.len(),
            total_tokens
        );

        Ok(selected)
    }

    /// Retrieve context using memory triggers from analysis
    pub async fn retrieve_by_triggers(
        &self,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<ContextObject>> {
        let mut all_results = Vec::new();

        for trigger in &analysis.memory_triggers {
            // Generate embedding for the search query
            let embeddings = self
                .embedding_provider
                .embed(&[trigger.search_query.clone()])
                .await?;

            if let Some(emb) = embeddings.into_iter().next() {
                let results = self.store.search_similar_objects(&emb, 3)?;
                for (obj, score) in results {
                    if score >= trigger.confidence {
                        all_results.push(obj);
                    }
                }
            }
        }

        // Deduplicate by ID
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|obj| seen.insert(obj.id));

        Ok(all_results)
    }

    /// Get context objects by type
    pub fn get_by_type(&self, object_type: ContextType) -> Result<Vec<ContextObject>> {
        self.store.get_by_type(object_type)
    }

    /// Get context objects by topic
    pub fn get_by_topic(&self, topic: &str) -> Result<Vec<ContextObject>> {
        self.store.get_by_topic(topic)
    }

    /// Get a specific context object by ID
    pub fn get(&self, id: &uuid::Uuid) -> Result<Option<ContextObject>> {
        self.store.get(id)
    }

    /// Remove a context object
    pub fn remove(&self, id: &uuid::Uuid) -> Result<Option<ContextObject>> {
        self.store.remove(id)
    }

    /// Clear all context
    pub fn clear(&self) -> Result<()> {
        self.store.clear()
    }

    /// Get the total number of stored context objects
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Flush to disk (if persistence is enabled)
    pub fn flush(&self) -> Result<()> {
        self.store.flush()
    }

    /// Update scoring weights
    pub fn set_scoring_weights(&mut self, weights: ScoringWeights) {
        self.scorer = RelevanceScorer::new()
            .with_weights(weights)
            .with_recency_half_life(self.config.recency_half_life_hours);
    }

    /// Summarize a context object (compress it)
    pub async fn summarize_object(
        &self,
        id: &uuid::Uuid,
        llm: Arc<dyn LlmProvider>,
    ) -> Result<()> {
        let obj = self.store.get(id)?.ok_or_else(|| {
            DpaError::ContextError(format!("Context object not found: {}", id))
        })?;

        if obj.summary.is_some() || obj.token_count < 100 {
            return Ok(()); // Already summarized or too short
        }

        let prompt = format!(
            "Summarize the following content in 1-2 sentences, preserving key information:\n\n{}",
            obj.content
        );

        let options = crate::llm::CompletionOptions {
            max_tokens: Some(200),
            temperature: Some(0.3),
            ..Default::default()
        };

        let response = llm.complete(&prompt, &options).await?;

        let mut updated = obj;
        updated.summary = Some(response.content);
        updated.is_compressed = true;

        self.store.update(updated)?;

        Ok(())
    }

    /// Get a summary of all stored context
    pub fn get_context_summary(&self) -> Result<ContextSummary> {
        let all = self.store.get_all()?;

        let mut type_counts = std::collections::HashMap::new();
        let mut total_tokens = 0;
        let mut topic_counts = std::collections::HashMap::new();

        for obj in &all {
            *type_counts.entry(format!("{:?}", obj.object_type)).or_insert(0) += 1;
            total_tokens += obj.token_count;

            for topic in &obj.topics {
                *topic_counts.entry(topic.clone()).or_insert(0) += 1;
            }
        }

        // Get top topics
        let mut topics: Vec<_> = topic_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));
        let top_topics: Vec<String> = topics.into_iter().take(10).map(|(t, _)| t).collect();

        Ok(ContextSummary {
            total_objects: all.len(),
            total_tokens,
            type_counts,
            top_topics,
        })
    }
}

/// Summary of stored context
#[derive(Debug, Clone)]
pub struct ContextSummary {
    pub total_objects: usize,
    pub total_tokens: usize,
    pub type_counts: std::collections::HashMap<String, usize>,
    pub top_topics: Vec<String>,
}

impl std::fmt::Debug for ContextEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextEngine")
            .field("store_size", &self.store.len())
            .field("session_id", &self.session_id)
            .field("turn_number", &self.turn_number)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require mocking LLM and embedding providers
    // See integration tests for complete testing
}
