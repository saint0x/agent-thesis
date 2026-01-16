//! Prompts for query analysis

/// System prompt for the query analyzer
pub const ANALYZER_SYSTEM_PROMPT: &str = r#"You are a query analysis assistant. Your task is to analyze user queries and extract structured information.

Analyze the query and provide a JSON response with the following structure:

{
  "sentiment": {
    "score": <float from -1.0 to 1.0>,
    "magnitude": <float from 0.0 to 1.0>,
    "label": "<very_negative|negative|neutral|positive|very_positive>"
  },
  "primary_intent": "<question|command|clarification|correction|continuation|new_topic|meta_query|statement|social|opinion|unknown>",
  "secondary_intents": ["<intent>"],
  "confidence": <float from 0.0 to 1.0>,
  "keywords": [
    {"text": "<keyword>", "weight": <float 0.0-1.0>, "is_technical": <bool>}
  ],
  "entities": [
    {"text": "<entity>", "entity_type": "<person|organization|location|date_time|number|technical_term|product_name|file_name|code_reference|url|other>"}
  ],
  "memory_triggers": [
    {
      "trigger_phrase": "<phrase that references past context>",
      "trigger_type": "<explicit|implicit|turn_reference|entity_reference>",
      "search_query": "<semantic query to find relevant context>",
      "confidence": <float 0.0-1.0>
    }
  ],
  "suggested_modules": [
    {"module_id": "<module>", "reason": "<why>", "confidence": <float 0.0-1.0>}
  ],
  "topics": ["<topic>"],
  "complexity": <float from 0.0 to 1.0>,
  "requires_knowledge": <bool>,
  "requires_reasoning": <bool>
}

Intent definitions:
- question: Asking for information or explanation
- command: Requesting a specific action to be performed
- clarification: Asking for more details about a previous response
- correction: Pointing out an error or mistake
- continuation: Following up on the same topic
- new_topic: Introducing a completely new subject
- meta_query: Question about the assistant itself
- statement: Providing information without asking
- social: Greeting, thanks, or social interaction
- opinion: Expressing a personal view

Module suggestions (suggest when relevant):
- coding_mode: Code-related queries
- analysis_mode: Data analysis or research
- creative_mode: Creative writing or brainstorming
- technical_mode: Technical explanations
- conversational_mode: Casual conversation
- memory_recall: When referencing past context

Memory trigger types:
- explicit: Direct reference like "remember when", "earlier", "as I said"
- implicit: Pronouns or references requiring context ("it", "that", "the same")
- turn_reference: Reference to specific past exchange
- entity_reference: Reference to entity mentioned earlier

Respond ONLY with valid JSON. No markdown, no explanation."#;

/// User prompt template for query analysis
pub fn analysis_prompt(query: &str, context_summary: Option<&str>) -> String {
    let mut prompt = format!("Analyze this query:\n\n\"{}\"\n", query);

    if let Some(summary) = context_summary {
        prompt.push_str(&format!(
            "\nConversation context summary:\n{}\n",
            summary
        ));
    }

    prompt.push_str("\nProvide the analysis as JSON:");
    prompt
}

/// Fallback analysis for when LLM fails
pub fn fallback_analysis(query: &str) -> super::types::QueryAnalysis {
    use super::types::*;

    let query_lower = query.to_lowercase();

    // Simple rule-based fallback
    let primary_intent = if query_lower.ends_with('?')
        || query_lower.starts_with("what")
        || query_lower.starts_with("how")
        || query_lower.starts_with("why")
        || query_lower.starts_with("when")
        || query_lower.starts_with("where")
        || query_lower.starts_with("who")
    {
        Intent::Question
    } else if query_lower.starts_with("please")
        || query_lower.starts_with("can you")
        || query_lower.starts_with("could you")
        || query_lower.contains("implement")
        || query_lower.contains("create")
        || query_lower.contains("make")
        || query_lower.contains("write")
    {
        Intent::Command
    } else if query_lower.contains("remember")
        || query_lower.contains("earlier")
        || query_lower.contains("before")
    {
        Intent::Continuation
    } else if query_lower.starts_with("hi")
        || query_lower.starts_with("hello")
        || query_lower.starts_with("thanks")
        || query_lower.starts_with("thank you")
    {
        Intent::Social
    } else {
        Intent::Statement
    };

    // Extract simple keywords (words longer than 3 chars)
    let keywords: Vec<Keyword> = query
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| Keyword {
            text: w.to_lowercase(),
            weight: 0.5,
            is_technical: false,
        })
        .take(5)
        .collect();

    // Detect memory triggers
    let memory_triggers = if query_lower.contains("remember")
        || query_lower.contains("earlier")
        || query_lower.contains("you said")
        || query_lower.contains("we discussed")
    {
        vec![MemoryTrigger {
            trigger_phrase: query.to_string(),
            trigger_type: MemoryTriggerType::Explicit,
            search_query: query.to_string(),
            confidence: 0.6,
        }]
    } else {
        Vec::new()
    };

    QueryAnalysis {
        original_query: query.to_string(),
        sentiment: Sentiment::default(),
        primary_intent,
        secondary_intents: Vec::new(),
        confidence: 0.4, // Low confidence for fallback
        keywords,
        entities: Vec::new(),
        memory_triggers,
        suggested_modules: Vec::new(),
        topics: Vec::new(),
        complexity: 0.5,
        requires_knowledge: primary_intent == Intent::Question,
        requires_reasoning: false,
        latency_ms: Some(0),
    }
}
