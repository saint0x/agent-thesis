//! Query Analyzer Module
//!
//! Uses a small, fast LLM to analyze user queries and extract:
//! - Sentiment (positive/negative/neutral with magnitude)
//! - Intent classification (question, command, clarification, etc.)
//! - Keywords and entities
//! - Memory triggers (references to past context)
//! - Suggested prompt modules to activate

pub mod types;
mod prompts;
mod analyzer;

pub use types::{
    QueryAnalysis, Intent, Sentiment, Keyword, Entity,
    MemoryTrigger, SuggestedModule, SentimentLabel,
};
pub use analyzer::QueryAnalyzer;
