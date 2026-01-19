//! Types for query analysis results

use serde::{Deserialize, Serialize};

/// Complete analysis of a user query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Original query text
    pub original_query: String,

    /// Sentiment analysis
    pub sentiment: Sentiment,

    /// Primary intent classification
    pub primary_intent: Intent,

    /// Secondary intents with lower confidence
    #[serde(default)]
    pub secondary_intents: Vec<Intent>,

    /// Overall confidence in the analysis
    pub confidence: f32,

    /// Extracted keywords with importance weights
    #[serde(default)]
    pub keywords: Vec<Keyword>,

    /// Named entities found in the query
    #[serde(default)]
    pub entities: Vec<Entity>,

    /// Detected references to past context
    #[serde(default)]
    pub memory_triggers: Vec<MemoryTrigger>,

    /// Suggested prompt modules to activate
    #[serde(default)]
    pub suggested_modules: Vec<SuggestedModule>,

    /// Topics detected in the query
    #[serde(default)]
    pub topics: Vec<String>,

    /// Estimated complexity (0.0 = simple, 1.0 = complex)
    #[serde(default)]
    pub complexity: f32,

    /// Whether the query requires factual knowledge
    #[serde(default)]
    pub requires_knowledge: bool,

    /// Whether the query requires reasoning
    #[serde(default)]
    pub requires_reasoning: bool,

    /// Analysis latency in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

impl Default for QueryAnalysis {
    fn default() -> Self {
        Self {
            original_query: String::new(),
            sentiment: Sentiment::default(),
            primary_intent: Intent::Question,
            secondary_intents: Vec::new(),
            confidence: 0.5,
            keywords: Vec::new(),
            entities: Vec::new(),
            memory_triggers: Vec::new(),
            suggested_modules: Vec::new(),
            topics: Vec::new(),
            complexity: 0.5,
            requires_knowledge: false,
            requires_reasoning: false,
            latency_ms: None,
        }
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentiment {
    /// Score from -1.0 (very negative) to 1.0 (very positive)
    pub score: f32,

    /// Magnitude from 0.0 (neutral) to 1.0 (strong emotion)
    pub magnitude: f32,

    /// Categorical label
    pub label: SentimentLabel,
}

impl Default for Sentiment {
    fn default() -> Self {
        Self {
            score: 0.0,
            magnitude: 0.0,
            label: SentimentLabel::Neutral,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SentimentLabel {
    VeryNegative,
    Negative,
    Neutral,
    Positive,
    VeryPositive,
}

/// Intent classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// Asking for information
    Question,

    /// Requesting an action
    Command,

    /// Asking for clarification on previous response
    Clarification,

    /// Correcting previous response or input
    Correction,

    /// Continuing previous topic
    Continuation,

    /// Starting a new topic
    NewTopic,

    /// Query about the agent itself
    MetaQuery,

    /// Providing information or context
    Statement,

    /// Greeting or social interaction
    Social,

    /// Expressing opinion or feedback
    Opinion,

    /// Unknown or ambiguous intent
    Unknown,
}

impl Default for Intent {
    fn default() -> Self {
        Intent::Unknown
    }
}

impl std::fmt::Display for Intent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Intent::Question => write!(f, "question"),
            Intent::Command => write!(f, "command"),
            Intent::Clarification => write!(f, "clarification"),
            Intent::Correction => write!(f, "correction"),
            Intent::Continuation => write!(f, "continuation"),
            Intent::NewTopic => write!(f, "new_topic"),
            Intent::MetaQuery => write!(f, "meta_query"),
            Intent::Statement => write!(f, "statement"),
            Intent::Social => write!(f, "social"),
            Intent::Opinion => write!(f, "opinion"),
            Intent::Unknown => write!(f, "unknown"),
        }
    }
}

/// Extracted keyword with importance weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyword {
    /// The keyword text
    pub text: String,

    /// Importance weight (0.0 to 1.0)
    pub weight: f32,

    /// Whether this is a technical term
    #[serde(default)]
    pub is_technical: bool,
}

/// Named entity extracted from the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text
    pub text: String,

    /// Entity type
    pub entity_type: EntityType,

    /// Start position in original query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,

    /// End position in original query
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Person,
    Organization,
    Location,
    DateTime,
    Number,
    TechnicalTerm,
    ProductName,
    FileName,
    CodeReference,
    Url,
    Other,
}

/// Detected reference to past context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrigger {
    /// The trigger phrase
    pub trigger_phrase: String,

    /// Type of memory reference
    pub trigger_type: MemoryTriggerType,

    /// Semantic search query to find relevant context
    pub search_query: String,

    /// Confidence in the trigger detection
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTriggerType {
    /// Explicit reference ("remember when...", "earlier you said...")
    Explicit,

    /// Implicit reference (pronouns, context-dependent terms)
    Implicit,

    /// Reference to a specific conversation turn
    TurnReference,

    /// Reference to a named entity from context
    EntityReference,
}

/// Suggested prompt module to activate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedModule {
    /// Module identifier
    pub module_id: String,

    /// Reason for suggesting this module
    pub reason: String,

    /// Confidence in the suggestion
    pub confidence: f32,
}
