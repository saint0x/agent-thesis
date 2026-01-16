//! Query Analyzer implementation

use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn, instrument};

use super::prompts::{ANALYZER_SYSTEM_PROMPT, analysis_prompt, fallback_analysis};
use super::types::QueryAnalysis;
use crate::llm::{LlmProvider, CompletionOptions};
use crate::{DpaError, Result};

/// Query analyzer that uses a small LLM for fast query analysis
pub struct QueryAnalyzer {
    /// Small, fast LLM for analysis
    llm: Arc<dyn LlmProvider>,

    /// Maximum tokens for analysis response
    max_tokens: usize,

    /// Temperature for analysis (lower = more deterministic)
    temperature: f32,
}

impl QueryAnalyzer {
    /// Create a new query analyzer with the given LLM provider
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            llm,
            max_tokens: 1024,
            temperature: 0.1, // Low temperature for consistent analysis
        }
    }

    /// Set the maximum tokens for analysis response
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature for analysis
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Analyze a user query
    #[instrument(skip(self, context_summary), fields(query_len = query.len()))]
    pub async fn analyze(
        &self,
        query: &str,
        context_summary: Option<&str>,
    ) -> Result<QueryAnalysis> {
        let start = Instant::now();

        let prompt = analysis_prompt(query, context_summary);

        let options = CompletionOptions {
            system: Some(ANALYZER_SYSTEM_PROMPT.to_string()),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            json_mode: true,
            stop_sequences: Vec::new(),
        };

        let response = match self.llm.complete(&prompt, &options).await {
            Ok(r) => r,
            Err(e) => {
                warn!("LLM analysis failed, using fallback: {}", e);
                let mut analysis = fallback_analysis(query);
                analysis.latency_ms = Some(start.elapsed().as_millis() as u64);
                return Ok(analysis);
            }
        };

        debug!("Raw analysis response: {}", response.content);

        // Parse the JSON response
        let mut analysis: QueryAnalysis = match serde_json::from_str(&response.content) {
            Ok(a) => a,
            Err(e) => {
                warn!("Failed to parse analysis JSON: {}", e);
                // Try to extract JSON from the response
                if let Some(json_str) = extract_json(&response.content) {
                    match serde_json::from_str(json_str) {
                        Ok(a) => a,
                        Err(_) => {
                            warn!("Failed to parse extracted JSON, using fallback");
                            fallback_analysis(query)
                        }
                    }
                } else {
                    warn!("No JSON found in response, using fallback");
                    fallback_analysis(query)
                }
            }
        };

        // Ensure original query is set
        analysis.original_query = query.to_string();
        analysis.latency_ms = Some(start.elapsed().as_millis() as u64);

        Ok(analysis)
    }

    /// Analyze multiple queries in batch
    pub async fn analyze_batch(
        &self,
        queries: &[String],
        context_summary: Option<&str>,
    ) -> Result<Vec<QueryAnalysis>> {
        let mut results = Vec::with_capacity(queries.len());

        for query in queries {
            results.push(self.analyze(query, context_summary).await?);
        }

        Ok(results)
    }

    /// Get quick intent classification without full analysis
    pub async fn quick_intent(&self, query: &str) -> Result<super::types::Intent> {
        // Use fallback rules for quick classification
        let analysis = fallback_analysis(query);
        Ok(analysis.primary_intent)
    }
}

/// Try to extract JSON from a response that might have extra text
fn extract_json(text: &str) -> Option<&str> {
    // Find the first { and last }
    let start = text.find('{')?;
    let end = text.rfind('}')?;

    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

impl std::fmt::Debug for QueryAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryAnalyzer")
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json() {
        let text = "Here is the analysis:\n{\"key\": \"value\"}\nDone.";
        assert_eq!(extract_json(text), Some("{\"key\": \"value\"}"));
    }

    #[test]
    fn test_extract_json_nested() {
        let text = "{\"outer\": {\"inner\": 1}}";
        assert_eq!(extract_json(text), Some("{\"outer\": {\"inner\": 1}}"));
    }

    #[test]
    fn test_fallback_question() {
        let analysis = fallback_analysis("What is the capital of France?");
        assert_eq!(analysis.primary_intent, super::super::types::Intent::Question);
    }

    #[test]
    fn test_fallback_command() {
        let analysis = fallback_analysis("Please implement a sorting algorithm");
        assert_eq!(analysis.primary_intent, super::super::types::Intent::Command);
    }
}
