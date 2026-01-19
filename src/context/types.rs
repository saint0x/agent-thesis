//! Types for the modular context engine

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A discrete, addressable unit of context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextObject {
    /// Unique identifier
    pub id: Uuid,

    /// The actual content
    pub content: String,

    /// Compressed/summarized version (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Type classification
    pub object_type: ContextType,

    /// Metadata about this context object
    pub metadata: ContextMetadata,

    /// Topics/tags for this context
    #[serde(default)]
    pub topics: Vec<String>,

    /// Importance score (0.0 to 1.0)
    #[serde(default = "default_importance")]
    pub importance: f32,

    /// Embedding vector (for similarity search)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,

    /// IDs of context objects this one references
    #[serde(default)]
    pub references: Vec<Uuid>,

    /// IDs of context objects that reference this one
    #[serde(default)]
    pub referenced_by: Vec<Uuid>,

    /// Whether this object is compressed
    #[serde(default)]
    pub is_compressed: bool,

    /// Original token count
    #[serde(default)]
    pub original_token_count: usize,

    /// Current token count (after compression if applicable)
    #[serde(default)]
    pub token_count: usize,
}

fn default_importance() -> f32 { 0.5 }

impl ContextObject {
    /// Create a new context object
    pub fn new(content: String, object_type: ContextType) -> Self {
        let token_count = content.len() / 4; // Rough estimate
        Self {
            id: Uuid::new_v4(),
            content,
            summary: None,
            object_type,
            metadata: ContextMetadata::new(),
            topics: Vec::new(),
            importance: 0.5,
            embedding: None,
            references: Vec::new(),
            referenced_by: Vec::new(),
            is_compressed: false,
            original_token_count: token_count,
            token_count,
        }
    }

    /// Create with specific ID
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Add topics
    pub fn with_topics(mut self, topics: Vec<String>) -> Self {
        self.topics = topics;
        self
    }

    /// Set importance
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add a reference to another context object
    pub fn add_reference(&mut self, other_id: Uuid) {
        if !self.references.contains(&other_id) {
            self.references.push(other_id);
        }
    }

    /// Mark this object as accessed
    pub fn mark_accessed(&mut self) {
        self.metadata.last_accessed = Utc::now();
        self.metadata.access_count += 1;
    }

    /// Get the content to use (summary if compressed, otherwise full content)
    pub fn active_content(&self) -> &str {
        if self.is_compressed {
            self.summary.as_deref().unwrap_or(&self.content)
        } else {
            &self.content
        }
    }
}

/// Type classification for context objects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextType {
    /// Standalone factual information
    Fact,

    /// User preferences, rules, instructions
    Instruction,

    /// Few-shot examples
    Example,

    /// Dialog history segment
    Conversation,

    /// Output from tool calls
    ToolResult,

    /// Application/system state
    SystemState,

    /// Code snippet or technical content
    Code,

    /// Document or article content
    Document,

    /// User profile information
    UserProfile,

    /// Task or goal description
    Task,

    /// Summary of multiple objects
    Summary,

    /// Other/uncategorized
    Other,
}

impl Default for ContextType {
    fn default() -> Self {
        ContextType::Other
    }
}

impl ContextType {
    /// Get a priority weight for this type (higher = more important to retrieve)
    pub fn priority_weight(&self) -> f32 {
        match self {
            ContextType::Instruction => 1.0,
            ContextType::UserProfile => 0.9,
            ContextType::Task => 0.85,
            ContextType::Fact => 0.8,
            ContextType::Code => 0.75,
            ContextType::Example => 0.7,
            ContextType::ToolResult => 0.65,
            ContextType::Conversation => 0.6,
            ContextType::Document => 0.55,
            ContextType::SystemState => 0.5,
            ContextType::Summary => 0.4,
            ContextType::Other => 0.3,
        }
    }
}

/// Metadata for a context object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// When this object was created
    pub created_at: DateTime<Utc>,

    /// When this object was last accessed
    pub last_accessed: DateTime<Utc>,

    /// Number of times this object has been accessed
    #[serde(default)]
    pub access_count: u32,

    /// Source of this context (e.g., "user_input", "tool:search", "partition")
    #[serde(default)]
    pub source: String,

    /// Session ID this context belongs to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Conversation turn number when created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_number: Option<usize>,

    /// Custom metadata fields
    #[serde(default)]
    pub custom: std::collections::HashMap<String, String>,
}

impl ContextMetadata {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
            source: String::new(),
            session_id: None,
            turn_number: None,
            custom: std::collections::HashMap::new(),
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }

    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    pub fn with_turn(mut self, turn: usize) -> Self {
        self.turn_number = Some(turn);
        self
    }
}

impl Default for ContextMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of partitioning raw context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResult {
    /// Generated context objects
    pub objects: Vec<ContextObject>,

    /// Whether the partitioning was lossless
    pub is_lossless: bool,

    /// Validation details
    pub validation: PartitionValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitionValidation {
    /// Original character count
    pub original_chars: usize,

    /// Total characters in partitions
    pub partition_chars: usize,

    /// Percentage of content preserved
    pub preservation_ratio: f32,

    /// Any warnings during partitioning
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_object_creation() {
        let co = ContextObject::new(
            "This is a test fact.".to_string(),
            ContextType::Fact,
        );

        assert!(!co.id.is_nil());
        assert_eq!(co.object_type, ContextType::Fact);
        assert_eq!(co.importance, 0.5);
    }

    #[test]
    fn test_context_object_builder() {
        let co = ContextObject::new("Test".to_string(), ContextType::Instruction)
            .with_topics(vec!["topic1".to_string()])
            .with_importance(0.9);

        assert_eq!(co.topics, vec!["topic1".to_string()]);
        assert_eq!(co.importance, 0.9);
    }

    #[test]
    fn test_mark_accessed() {
        let mut co = ContextObject::new("Test".to_string(), ContextType::Fact);
        let initial_access = co.metadata.last_accessed;

        std::thread::sleep(std::time::Duration::from_millis(10));
        co.mark_accessed();

        assert!(co.metadata.last_accessed > initial_access);
        assert_eq!(co.metadata.access_count, 1);
    }
}
