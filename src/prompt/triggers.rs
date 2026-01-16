//! Trigger system for conditional prompt module activation

use serde::{Deserialize, Serialize};
use regex::Regex;

use crate::analyzer::{QueryAnalysis, Intent};

/// A trigger condition for activating a prompt module
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    /// Activate when any of the keywords are present
    KeywordPresent {
        keywords: Vec<String>,
        #[serde(default)]
        case_sensitive: bool,
    },

    /// Activate when none of the keywords are present
    KeywordAbsent {
        keywords: Vec<String>,
        #[serde(default)]
        case_sensitive: bool,
    },

    /// Activate when a regex pattern matches
    PatternMatch {
        pattern: String,
        #[serde(default)]
        case_sensitive: bool,
    },

    /// Activate when sentiment is in range
    SentimentRange {
        min: f32,
        max: f32,
    },

    /// Activate when sentiment magnitude is in range
    SentimentMagnitudeRange {
        min: f32,
        max: f32,
    },

    /// Activate when intent matches
    IntentMatch {
        intents: Vec<String>,
    },

    /// Activate when any topic is present
    TopicPresent {
        topics: Vec<String>,
    },

    /// Activate when complexity is in range
    ComplexityRange {
        min: f32,
        max: f32,
    },

    /// Activate when query requires knowledge
    RequiresKnowledge,

    /// Activate when query requires reasoning
    RequiresReasoning,

    /// Activate when memory triggers are detected
    MemoryTriggerDetected {
        #[serde(default)]
        min_confidence: f32,
    },

    /// Activate when suggested by the analyzer
    SuggestedByAnalyzer {
        module_id: String,
        #[serde(default)]
        min_confidence: f32,
    },

    /// Activate when conversation length is in range
    ConversationLength {
        min: usize,
        #[serde(default)]
        max: Option<usize>,
    },

    /// Always activate
    Always,

    /// Never activate (for disabled modules)
    Never,

    /// Boolean AND of multiple triggers
    And {
        triggers: Vec<Trigger>,
    },

    /// Boolean OR of multiple triggers
    Or {
        triggers: Vec<Trigger>,
    },

    /// Boolean NOT of a trigger
    Not {
        trigger: Box<Trigger>,
    },
}

/// Evaluator for trigger conditions
pub struct TriggerEvaluator {
    /// Number of conversation turns (for ConversationLength trigger)
    conversation_length: usize,
}

impl TriggerEvaluator {
    /// Create a new trigger evaluator
    pub fn new() -> Self {
        Self {
            conversation_length: 0,
        }
    }

    /// Set the conversation length
    pub fn with_conversation_length(mut self, length: usize) -> Self {
        self.conversation_length = length;
        self
    }

    /// Evaluate a trigger against a query analysis
    pub fn evaluate(&self, trigger: &Trigger, analysis: &QueryAnalysis) -> bool {
        match trigger {
            Trigger::KeywordPresent { keywords, case_sensitive } => {
                let query = if *case_sensitive {
                    analysis.original_query.clone()
                } else {
                    analysis.original_query.to_lowercase()
                };

                keywords.iter().any(|kw| {
                    let kw = if *case_sensitive { kw.clone() } else { kw.to_lowercase() };
                    query.contains(&kw)
                })
            }

            Trigger::KeywordAbsent { keywords, case_sensitive } => {
                let query = if *case_sensitive {
                    analysis.original_query.clone()
                } else {
                    analysis.original_query.to_lowercase()
                };

                !keywords.iter().any(|kw| {
                    let kw = if *case_sensitive { kw.clone() } else { kw.to_lowercase() };
                    query.contains(&kw)
                })
            }

            Trigger::PatternMatch { pattern, case_sensitive } => {
                let regex = if *case_sensitive {
                    Regex::new(pattern)
                } else {
                    Regex::new(&format!("(?i){}", pattern))
                };

                match regex {
                    Ok(re) => re.is_match(&analysis.original_query),
                    Err(_) => false,
                }
            }

            Trigger::SentimentRange { min, max } => {
                analysis.sentiment.score >= *min && analysis.sentiment.score <= *max
            }

            Trigger::SentimentMagnitudeRange { min, max } => {
                analysis.sentiment.magnitude >= *min && analysis.sentiment.magnitude <= *max
            }

            Trigger::IntentMatch { intents } => {
                let primary = analysis.primary_intent.to_string();
                intents.iter().any(|i| i == &primary)
            }

            Trigger::TopicPresent { topics } => {
                topics.iter().any(|t| {
                    analysis.topics.iter().any(|at| {
                        at.to_lowercase().contains(&t.to_lowercase())
                    })
                })
            }

            Trigger::ComplexityRange { min, max } => {
                analysis.complexity >= *min && analysis.complexity <= *max
            }

            Trigger::RequiresKnowledge => analysis.requires_knowledge,

            Trigger::RequiresReasoning => analysis.requires_reasoning,

            Trigger::MemoryTriggerDetected { min_confidence } => {
                analysis.memory_triggers.iter().any(|mt| mt.confidence >= *min_confidence)
            }

            Trigger::SuggestedByAnalyzer { module_id, min_confidence } => {
                analysis.suggested_modules.iter().any(|sm| {
                    &sm.module_id == module_id && sm.confidence >= *min_confidence
                })
            }

            Trigger::ConversationLength { min, max } => {
                let in_min = self.conversation_length >= *min;
                let in_max = max.map_or(true, |m| self.conversation_length <= m);
                in_min && in_max
            }

            Trigger::Always => true,

            Trigger::Never => false,

            Trigger::And { triggers } => {
                triggers.iter().all(|t| self.evaluate(t, analysis))
            }

            Trigger::Or { triggers } => {
                triggers.iter().any(|t| self.evaluate(t, analysis))
            }

            Trigger::Not { trigger } => {
                !self.evaluate(trigger, analysis)
            }
        }
    }

    /// Evaluate all triggers for a module (OR logic - any trigger activates)
    pub fn evaluate_module_triggers(
        &self,
        triggers: &[Trigger],
        analysis: &QueryAnalysis,
    ) -> bool {
        if triggers.is_empty() {
            return false; // No triggers = never activate
        }

        triggers.iter().any(|t| self.evaluate(t, analysis))
    }
}

impl Default for TriggerEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::types::*;

    fn mock_analysis(query: &str) -> QueryAnalysis {
        QueryAnalysis {
            original_query: query.to_string(),
            sentiment: Sentiment {
                score: 0.5,
                magnitude: 0.3,
                label: SentimentLabel::Positive,
            },
            primary_intent: Intent::Question,
            topics: vec!["programming".to_string(), "rust".to_string()],
            complexity: 0.6,
            requires_knowledge: true,
            requires_reasoning: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_keyword_present() {
        let evaluator = TriggerEvaluator::new();
        let analysis = mock_analysis("How do I implement a function in Rust?");

        let trigger = Trigger::KeywordPresent {
            keywords: vec!["implement".to_string(), "function".to_string()],
            case_sensitive: false,
        };

        assert!(evaluator.evaluate(&trigger, &analysis));
    }

    #[test]
    fn test_keyword_absent() {
        let evaluator = TriggerEvaluator::new();
        let analysis = mock_analysis("How do I implement a function?");

        let trigger = Trigger::KeywordAbsent {
            keywords: vec!["python".to_string(), "java".to_string()],
            case_sensitive: false,
        };

        assert!(evaluator.evaluate(&trigger, &analysis));
    }

    #[test]
    fn test_intent_match() {
        let evaluator = TriggerEvaluator::new();
        let analysis = mock_analysis("What is Rust?");

        let trigger = Trigger::IntentMatch {
            intents: vec!["question".to_string(), "command".to_string()],
        };

        assert!(evaluator.evaluate(&trigger, &analysis));
    }

    #[test]
    fn test_and_trigger() {
        let evaluator = TriggerEvaluator::new();
        let analysis = mock_analysis("How do I implement a function?");

        let trigger = Trigger::And {
            triggers: vec![
                Trigger::KeywordPresent {
                    keywords: vec!["implement".to_string()],
                    case_sensitive: false,
                },
                Trigger::IntentMatch {
                    intents: vec!["question".to_string()],
                },
            ],
        };

        assert!(evaluator.evaluate(&trigger, &analysis));
    }

    #[test]
    fn test_not_trigger() {
        let evaluator = TriggerEvaluator::new();
        let analysis = mock_analysis("Hello there!");

        let trigger = Trigger::Not {
            trigger: Box::new(Trigger::KeywordPresent {
                keywords: vec!["implement".to_string()],
                case_sensitive: false,
            }),
        };

        assert!(evaluator.evaluate(&trigger, &analysis));
    }
}
