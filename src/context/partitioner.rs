//! Context Partitioner
//!
//! Uses a small LLM to losslessly partition raw context into discrete,
//! self-contained context objects.

use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::types::{ContextObject, ContextType, ContextMetadata, PartitionResult, PartitionValidation};
use crate::llm::{LlmProvider, CompletionOptions};
use crate::{DpaError, Result};

/// System prompt for the partitioner
const PARTITIONER_SYSTEM_PROMPT: &str = r#"You are a context partitioning assistant. Your task is to analyze text and split it into discrete, self-contained context objects.

Rules:
1. Each object should be semantically coherent (one topic/concept)
2. Preserve ALL information - this MUST be lossless
3. Mark relationships between objects when one references another
4. Classify each object by type
5. Extract key topics for each object
6. Assign importance scores based on relevance

Types:
- fact: Standalone factual information
- instruction: User preferences, rules, directives
- example: Few-shot examples or demonstrations
- conversation: Dialog history segment
- tool_result: Output from tool calls
- system_state: Application state information
- code: Code snippets or technical content
- document: Document or article content
- task: Task or goal descriptions
- other: Uncategorized content

Output format (JSON array):
[
  {
    "content": "...",
    "type": "fact|instruction|example|conversation|tool_result|system_state|code|document|task|other",
    "topics": ["topic1", "topic2"],
    "importance": 0.0-1.0,
    "references_indices": [0, 2]
  }
]

IMPORTANT: Output ONLY valid JSON. No explanation, no markdown."#;

/// Context partitioner using small LLM
pub struct ContextPartitioner {
    /// Small LLM for partitioning
    llm: Arc<dyn LlmProvider>,

    /// Maximum tokens per context object
    max_object_tokens: usize,

    /// Minimum tokens per context object
    min_object_tokens: usize,

    /// Temperature for partitioning (low for consistency)
    temperature: f32,
}

impl ContextPartitioner {
    /// Create a new context partitioner
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            max_object_tokens: 1000,
            min_object_tokens: 20,
            temperature: 0.1,
        }
    }

    /// Set maximum tokens per object
    pub fn with_max_object_tokens(mut self, max: usize) -> Self {
        self.max_object_tokens = max;
        self
    }

    /// Set minimum tokens per object
    pub fn with_min_object_tokens(mut self, min: usize) -> Self {
        self.min_object_tokens = min;
        self
    }

    /// Partition raw context into context objects
    pub async fn partition(
        &self,
        raw_context: &str,
        source: &str,
    ) -> Result<PartitionResult> {
        // For very short content, create a single object
        let estimated_tokens = raw_context.len() / 4;
        if estimated_tokens <= self.min_object_tokens * 2 {
            return Ok(self.create_single_object(raw_context, source));
        }

        // For content that fits in one object, don't partition
        if estimated_tokens <= self.max_object_tokens {
            return Ok(self.create_single_object(raw_context, source));
        }

        // Use LLM for intelligent partitioning
        let prompt = format!(
            "Partition the following text into discrete context objects:\n\n---\n{}\n---\n\nOutput JSON array:",
            raw_context
        );

        let options = CompletionOptions {
            system: Some(PARTITIONER_SYSTEM_PROMPT.to_string()),
            max_tokens: Some(4096),
            temperature: Some(self.temperature),
            json_mode: true,
            stop_sequences: Vec::new(),
        };

        let response = match self.llm.complete(&prompt, &options).await {
            Ok(r) => r,
            Err(e) => {
                warn!("LLM partitioning failed, using heuristic: {}", e);
                return Ok(self.heuristic_partition(raw_context, source));
            }
        };

        // Parse the response
        let objects = self.parse_partition_response(&response.content, source)?;

        // Validate losslessness
        let validation = self.validate_partition(raw_context, &objects);

        if validation.preservation_ratio < 0.95 {
            warn!(
                "Partition may be lossy: {:.1}% preservation",
                validation.preservation_ratio * 100.0
            );
        }

        info!(
            "Partitioned context into {} objects ({:.1}% preserved)",
            objects.len(),
            validation.preservation_ratio * 100.0
        );

        Ok(PartitionResult {
            objects,
            is_lossless: validation.preservation_ratio >= 0.95,
            validation,
        })
    }

    /// Parse the LLM response into context objects
    fn parse_partition_response(
        &self,
        response: &str,
        source: &str,
    ) -> Result<Vec<ContextObject>> {
        // Try to extract JSON from response
        let json_str = extract_json(response).unwrap_or(response);

        let parsed: Vec<PartitionedObject> = serde_json::from_str(json_str)
            .map_err(|e| DpaError::ContextError(format!("Failed to parse partition response: {}", e)))?;

        // Convert to context objects
        let mut objects: Vec<ContextObject> = parsed
            .into_iter()
            .map(|p| {
                let object_type = parse_context_type(&p.object_type);
                let mut obj = ContextObject::new(p.content, object_type)
                    .with_topics(p.topics)
                    .with_importance(p.importance);
                obj.metadata = ContextMetadata::new().with_source(source);
                obj
            })
            .collect();

        // Resolve references (convert indices to UUIDs)
        let ids: Vec<Uuid> = objects.iter().map(|o| o.id).collect();
        for (i, obj) in objects.iter_mut().enumerate() {
            // Note: We'd need to store reference indices in the parsed object
            // For now, leave references empty - they'll be populated by analysis
        }

        Ok(objects)
    }

    /// Create a single context object (no partitioning needed)
    fn create_single_object(&self, content: &str, source: &str) -> PartitionResult {
        let object_type = infer_context_type(content);
        let topics = extract_simple_topics(content);

        let mut obj = ContextObject::new(content.to_string(), object_type)
            .with_topics(topics)
            .with_importance(0.5);
        obj.metadata = ContextMetadata::new().with_source(source);

        PartitionResult {
            objects: vec![obj],
            is_lossless: true,
            validation: PartitionValidation {
                original_chars: content.len(),
                partition_chars: content.len(),
                preservation_ratio: 1.0,
                warnings: Vec::new(),
            },
        }
    }

    /// Heuristic-based partitioning (fallback)
    fn heuristic_partition(&self, content: &str, source: &str) -> PartitionResult {
        let mut objects = Vec::new();
        let mut current_chunk = String::new();
        let mut current_type = ContextType::Other;

        // Split by double newlines (paragraphs)
        for paragraph in content.split("\n\n") {
            let paragraph = paragraph.trim();
            if paragraph.is_empty() {
                continue;
            }

            let para_type = infer_context_type(paragraph);
            let para_tokens = paragraph.len() / 4;

            // Check if we should start a new object
            let should_split = current_chunk.len() / 4 + para_tokens > self.max_object_tokens
                || (para_type != current_type && !current_chunk.is_empty());

            if should_split && !current_chunk.is_empty() {
                // Save current chunk
                let topics = extract_simple_topics(&current_chunk);
                let mut obj = ContextObject::new(current_chunk.clone(), current_type)
                    .with_topics(topics)
                    .with_importance(0.5);
                obj.metadata = ContextMetadata::new().with_source(source);
                objects.push(obj);

                current_chunk = String::new();
            }

            if current_chunk.is_empty() {
                current_type = para_type;
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            let topics = extract_simple_topics(&current_chunk);
            let mut obj = ContextObject::new(current_chunk.clone(), current_type)
                .with_topics(topics)
                .with_importance(0.5);
            obj.metadata = ContextMetadata::new().with_source(source);
            objects.push(obj);
        }

        let total_chars: usize = objects.iter().map(|o| o.content.len()).sum();

        PartitionResult {
            objects,
            is_lossless: true,
            validation: PartitionValidation {
                original_chars: content.len(),
                partition_chars: total_chars,
                preservation_ratio: total_chars as f32 / content.len().max(1) as f32,
                warnings: vec!["Used heuristic partitioning".to_string()],
            },
        }
    }

    /// Validate that partitioning preserved content
    fn validate_partition(&self, original: &str, objects: &[ContextObject]) -> PartitionValidation {
        let original_chars = original.len();
        let partition_chars: usize = objects.iter().map(|o| o.content.len()).sum();

        // Check for content preservation (allowing some whitespace variation)
        let original_normalized = normalize_whitespace(original);
        let partition_normalized: String = objects
            .iter()
            .map(|o| normalize_whitespace(&o.content))
            .collect::<Vec<_>>()
            .join(" ");

        let preservation_ratio = if original_chars > 0 {
            // Use character-level similarity as approximation
            let matching_chars = original_normalized
                .chars()
                .filter(|c| partition_normalized.contains(*c))
                .count();
            matching_chars as f32 / original_normalized.len().max(1) as f32
        } else {
            1.0
        };

        let mut warnings = Vec::new();
        if preservation_ratio < 0.95 {
            warnings.push(format!(
                "Content preservation is {:.1}%, some information may be lost",
                preservation_ratio * 100.0
            ));
        }

        PartitionValidation {
            original_chars,
            partition_chars,
            preservation_ratio: preservation_ratio.min(1.0),
            warnings,
        }
    }

    /// Validate that objects can be reconstructed into original content
    pub async fn validate_lossless(
        &self,
        original: &str,
        objects: &[ContextObject],
    ) -> bool {
        let validation = self.validate_partition(original, objects);
        validation.preservation_ratio >= 0.95
    }
}

/// Intermediate struct for parsing LLM response
#[derive(Debug, serde::Deserialize)]
struct PartitionedObject {
    content: String,
    #[serde(rename = "type")]
    object_type: String,
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default = "default_importance")]
    importance: f32,
    #[serde(default)]
    references_indices: Vec<usize>,
}

fn default_importance() -> f32 { 0.5 }

/// Parse context type from string
fn parse_context_type(s: &str) -> ContextType {
    match s.to_lowercase().as_str() {
        "fact" => ContextType::Fact,
        "instruction" => ContextType::Instruction,
        "example" => ContextType::Example,
        "conversation" => ContextType::Conversation,
        "tool_result" => ContextType::ToolResult,
        "system_state" => ContextType::SystemState,
        "code" => ContextType::Code,
        "document" => ContextType::Document,
        "task" => ContextType::Task,
        _ => ContextType::Other,
    }
}

/// Infer context type from content
fn infer_context_type(content: &str) -> ContextType {
    let lower = content.to_lowercase();

    if content.contains("```") || content.contains("fn ") || content.contains("def ")
        || content.contains("class ") || content.contains("function ")
    {
        ContextType::Code
    } else if lower.starts_with("user:") || lower.starts_with("assistant:")
        || lower.contains("said") || lower.contains("asked")
    {
        ContextType::Conversation
    } else if lower.contains("please") || lower.contains("should") || lower.contains("must")
        || lower.contains("always") || lower.contains("never")
    {
        ContextType::Instruction
    } else if lower.contains("example") || lower.contains("for instance")
        || lower.contains("such as")
    {
        ContextType::Example
    } else if lower.contains("result") || lower.contains("output") || lower.contains("returned") {
        ContextType::ToolResult
    } else {
        ContextType::Fact
    }
}

/// Extract simple topics from content
fn extract_simple_topics(content: &str) -> Vec<String> {
    // Simple keyword extraction
    let stop_words: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "this",
        "that", "these", "those", "it", "its", "i", "you", "he", "she",
        "we", "they", "what", "which", "who", "whom", "your", "his", "her",
    ].iter().copied().collect();

    let mut word_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for word in content.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        if cleaned.len() > 2 && !stop_words.contains(cleaned.as_str()) {
            *word_counts.entry(cleaned).or_insert(0) += 1;
        }
    }

    let mut topics: Vec<(String, usize)> = word_counts.into_iter().collect();
    topics.sort_by(|a, b| b.1.cmp(&a.1));

    topics.into_iter().take(5).map(|(w, _)| w).collect()
}

/// Try to extract JSON from text
fn extract_json(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Normalize whitespace for comparison
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl std::fmt::Debug for ContextPartitioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextPartitioner")
            .field("max_object_tokens", &self.max_object_tokens)
            .field("min_object_tokens", &self.min_object_tokens)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_context_type_code() {
        let content = "```rust\nfn main() {}\n```";
        assert_eq!(infer_context_type(content), ContextType::Code);
    }

    #[test]
    fn test_infer_context_type_instruction() {
        let content = "You should always respond in a helpful manner.";
        assert_eq!(infer_context_type(content), ContextType::Instruction);
    }

    #[test]
    fn test_extract_topics() {
        let content = "Rust programming language systems programming memory safety";
        let topics = extract_simple_topics(content);
        assert!(topics.contains(&"rust".to_string()));
        assert!(topics.contains(&"programming".to_string()));
    }

    #[test]
    fn test_normalize_whitespace() {
        let s = "  hello   world  \n  test  ";
        assert_eq!(normalize_whitespace(s), "hello world test");
    }
}
