//! Relevance scoring for context retrieval

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::types::{ContextObject, ContextType};
use crate::embeddings::cosine_similarity;

/// Weights for relevance scoring components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for semantic similarity (cosine similarity of embeddings)
    pub semantic: f32,

    /// Weight for recency (how recently the object was accessed)
    pub recency: f32,

    /// Weight for reference importance (how many other objects reference this)
    pub reference: f32,

    /// Weight for type match (how well the object type matches query requirements)
    pub type_match: f32,

    /// Weight for topic overlap (shared topics between query and object)
    pub topic_overlap: f32,

    /// Weight for intrinsic importance (object's own importance score)
    pub importance: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            semantic: 0.35,
            recency: 0.15,
            reference: 0.10,
            type_match: 0.15,
            topic_overlap: 0.15,
            importance: 0.10,
        }
    }
}

impl ScoringWeights {
    /// Create weights optimized for factual retrieval
    pub fn factual() -> Self {
        Self {
            semantic: 0.45,
            recency: 0.05,
            reference: 0.15,
            type_match: 0.15,
            topic_overlap: 0.15,
            importance: 0.05,
        }
    }

    /// Create weights optimized for conversational context
    pub fn conversational() -> Self {
        Self {
            semantic: 0.25,
            recency: 0.30,
            reference: 0.10,
            type_match: 0.10,
            topic_overlap: 0.15,
            importance: 0.10,
        }
    }

    /// Create weights optimized for instruction following
    pub fn instructional() -> Self {
        Self {
            semantic: 0.20,
            recency: 0.05,
            reference: 0.10,
            type_match: 0.30,
            topic_overlap: 0.15,
            importance: 0.20,
        }
    }

    /// Validate that weights sum to 1.0
    pub fn validate(&self) -> bool {
        let sum = self.semantic + self.recency + self.reference
            + self.type_match + self.topic_overlap + self.importance;
        (sum - 1.0).abs() < 0.01
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.semantic + self.recency + self.reference
            + self.type_match + self.topic_overlap + self.importance;
        if sum > 0.0 {
            self.semantic /= sum;
            self.recency /= sum;
            self.reference /= sum;
            self.type_match /= sum;
            self.topic_overlap /= sum;
            self.importance /= sum;
        }
    }
}

/// Relevance scorer for context objects
pub struct RelevanceScorer {
    /// Scoring weights
    weights: ScoringWeights,

    /// Half-life for recency decay (in hours)
    recency_half_life_hours: f64,

    /// Required context types (from query analysis)
    required_types: Vec<ContextType>,
}

impl RelevanceScorer {
    /// Create a new relevance scorer with default weights
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
            recency_half_life_hours: 24.0,
            required_types: Vec::new(),
        }
    }

    /// Create with custom weights
    pub fn with_weights(mut self, weights: ScoringWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the recency half-life
    pub fn with_recency_half_life(mut self, hours: f64) -> Self {
        self.recency_half_life_hours = hours;
        self
    }

    /// Set required context types
    pub fn with_required_types(mut self, types: Vec<ContextType>) -> Self {
        self.required_types = types;
        self
    }

    /// Score a context object for relevance to a query
    pub fn score(
        &self,
        object: &ContextObject,
        query_embedding: Option<&[f32]>,
        query_topics: &[String],
        now: DateTime<Utc>,
    ) -> f32 {
        let mut score = 0.0;

        // Semantic similarity
        if let (Some(obj_emb), Some(query_emb)) = (&object.embedding, query_embedding) {
            let semantic_score = cosine_similarity(obj_emb, query_emb);
            score += self.weights.semantic * semantic_score.max(0.0);
        }

        // Recency score
        let recency_score = self.calculate_recency_score(object.metadata.last_accessed, now);
        score += self.weights.recency * recency_score;

        // Reference importance
        let reference_score = self.calculate_reference_score(object);
        score += self.weights.reference * reference_score;

        // Type match
        let type_score = self.calculate_type_score(object.object_type);
        score += self.weights.type_match * type_score;

        // Topic overlap
        let topic_score = self.calculate_topic_overlap(&object.topics, query_topics);
        score += self.weights.topic_overlap * topic_score;

        // Intrinsic importance
        score += self.weights.importance * object.importance;

        score.clamp(0.0, 1.0)
    }

    /// Score multiple objects and return sorted by relevance
    pub fn rank(
        &self,
        objects: &[ContextObject],
        query_embedding: Option<&[f32]>,
        query_topics: &[String],
    ) -> Vec<(usize, f32)> {
        let now = Utc::now();

        let mut scored: Vec<(usize, f32)> = objects
            .iter()
            .enumerate()
            .map(|(i, obj)| {
                let score = self.score(obj, query_embedding, query_topics, now);
                (i, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }

    /// Calculate recency score using exponential decay
    fn calculate_recency_score(&self, last_accessed: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
        let hours_since_access = (now - last_accessed).num_seconds() as f64 / 3600.0;

        if hours_since_access <= 0.0 {
            return 1.0;
        }

        // Exponential decay: score = 0.5^(hours / half_life)
        let decay = 0.5_f64.powf(hours_since_access / self.recency_half_life_hours);
        decay as f32
    }

    /// Calculate reference importance score
    fn calculate_reference_score(&self, object: &ContextObject) -> f32 {
        // More references = more important
        // Use logarithmic scaling to prevent dominance
        let ref_count = object.referenced_by.len() as f32;
        if ref_count == 0.0 {
            return 0.0;
        }

        // log2(ref_count + 1) / log2(max_expected_refs + 1)
        let max_expected: f32 = 10.0;
        ((ref_count + 1.0).log2() / (max_expected + 1.0).log2()).min(1.0)
    }

    /// Calculate type match score
    fn calculate_type_score(&self, object_type: ContextType) -> f32 {
        if self.required_types.is_empty() {
            // No specific requirements, use type's intrinsic priority
            return object_type.priority_weight();
        }

        // Check if type matches requirements
        if self.required_types.contains(&object_type) {
            1.0
        } else {
            // Partial credit based on type priority
            object_type.priority_weight() * 0.5
        }
    }

    /// Calculate topic overlap score
    fn calculate_topic_overlap(&self, object_topics: &[String], query_topics: &[String]) -> f32 {
        if object_topics.is_empty() || query_topics.is_empty() {
            return 0.0;
        }

        let object_topics_lower: std::collections::HashSet<String> = object_topics
            .iter()
            .map(|t| t.to_lowercase())
            .collect();

        let query_topics_lower: std::collections::HashSet<String> = query_topics
            .iter()
            .map(|t| t.to_lowercase())
            .collect();

        let intersection = object_topics_lower
            .intersection(&query_topics_lower)
            .count() as f32;

        let union = object_topics_lower
            .union(&query_topics_lower)
            .count() as f32;

        if union == 0.0 {
            0.0
        } else {
            intersection / union // Jaccard similarity
        }
    }
}

impl Default for RelevanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RelevanceScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelevanceScorer")
            .field("weights", &self.weights)
            .field("recency_half_life_hours", &self.recency_half_life_hours)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::types::ContextMetadata;

    fn create_test_object(importance: f32, topics: Vec<String>) -> ContextObject {
        ContextObject {
            id: uuid::Uuid::new_v4(),
            content: "Test content".to_string(),
            summary: None,
            object_type: ContextType::Fact,
            metadata: ContextMetadata::new(),
            topics,
            importance,
            embedding: None,
            references: Vec::new(),
            referenced_by: Vec::new(),
            is_compressed: false,
            original_token_count: 10,
            token_count: 10,
        }
    }

    #[test]
    fn test_recency_score() {
        let scorer = RelevanceScorer::new().with_recency_half_life(24.0);
        let now = Utc::now();

        // Just accessed
        let score = scorer.calculate_recency_score(now, now);
        assert!((score - 1.0).abs() < 0.01);

        // Accessed 24 hours ago (half-life)
        let yesterday = now - chrono::Duration::hours(24);
        let score = scorer.calculate_recency_score(yesterday, now);
        assert!((score - 0.5).abs() < 0.01);

        // Accessed 48 hours ago (two half-lives)
        let two_days = now - chrono::Duration::hours(48);
        let score = scorer.calculate_recency_score(two_days, now);
        assert!((score - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_topic_overlap() {
        let scorer = RelevanceScorer::new();

        // Full overlap
        let obj_topics = vec!["rust".to_string(), "programming".to_string()];
        let query_topics = vec!["rust".to_string(), "programming".to_string()];
        let score = scorer.calculate_topic_overlap(&obj_topics, &query_topics);
        assert!((score - 1.0).abs() < 0.01);

        // No overlap
        let obj_topics = vec!["rust".to_string()];
        let query_topics = vec!["python".to_string()];
        let score = scorer.calculate_topic_overlap(&obj_topics, &query_topics);
        assert!((score - 0.0).abs() < 0.01);

        // Partial overlap
        let obj_topics = vec!["rust".to_string(), "systems".to_string()];
        let query_topics = vec!["rust".to_string(), "web".to_string()];
        let score = scorer.calculate_topic_overlap(&obj_topics, &query_topics);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_weights_validation() {
        let weights = ScoringWeights::default();
        assert!(weights.validate());
    }

    #[test]
    fn test_ranking() {
        let scorer = RelevanceScorer::new();

        let obj1 = create_test_object(0.9, vec!["rust".to_string()]);
        let obj2 = create_test_object(0.3, vec!["python".to_string()]);
        let obj3 = create_test_object(0.6, vec!["rust".to_string(), "systems".to_string()]);

        let objects = vec![obj1, obj2, obj3];
        let query_topics = vec!["rust".to_string()];

        let ranked = scorer.rank(&objects, None, &query_topics);

        // obj1 or obj3 should be first (both have "rust" topic)
        assert!(ranked[0].0 == 0 || ranked[0].0 == 2);
        // obj2 should be last (no matching topic, low importance)
        assert_eq!(ranked[2].0, 1);
    }
}
