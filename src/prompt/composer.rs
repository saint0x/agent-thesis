//! Prompt composition engine

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

use super::templates::{TemplateEngine, PromptContext, AnalysisContext, ContextObjectContext};
use super::triggers::TriggerEvaluator;
use super::types::{
    ComposedPrompt, CompositionMetadata, Example, ModuleConfig, PromptModule, PromptSection,
};
use crate::analyzer::QueryAnalysis;
use crate::context::ContextObject;
use crate::{DpaError, Result};

/// Dynamic prompt builder
pub struct PromptBuilder {
    /// Registered prompt modules
    modules: HashMap<String, PromptModule>,

    /// Template engine for rendering
    template_engine: TemplateEngine,

    /// Trigger evaluator
    trigger_evaluator: TriggerEvaluator,

    /// Configuration
    config: ModuleConfig,

    /// Example library for few-shot learning
    examples: Vec<Example>,
}

impl PromptBuilder {
    /// Create a new prompt builder with configuration
    pub fn new(config: ModuleConfig) -> Self {
        Self {
            modules: HashMap::new(),
            template_engine: TemplateEngine::new(),
            trigger_evaluator: TriggerEvaluator::new(),
            config,
            examples: Vec::new(),
        }
    }

    /// Set the conversation length for trigger evaluation
    pub fn with_conversation_length(mut self, length: usize) -> Self {
        self.trigger_evaluator = self.trigger_evaluator.with_conversation_length(length);
        self
    }

    /// Register a prompt module
    pub fn register_module(&mut self, module: PromptModule) {
        debug!("Registering module: {}", module.id);
        self.modules.insert(module.id.clone(), module);
    }

    /// Register multiple modules
    pub fn register_modules(&mut self, modules: Vec<PromptModule>) {
        for module in modules {
            self.register_module(module);
        }
    }

    /// Load modules from a TOML file
    pub fn load_modules_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| DpaError::IoError(e))?;

        let modules: ModulesFile = toml::from_str(&content)
            .map_err(|e| DpaError::ConfigError(format!("Failed to parse modules file: {}", e)))?;

        let module_count = modules.modules.len();
        for module in modules.modules {
            self.register_module(module);
        }

        info!("Loaded {} modules from file", module_count);
        Ok(())
    }

    /// Add examples for few-shot learning
    pub fn add_examples(&mut self, examples: Vec<Example>) {
        self.examples.extend(examples);
    }

    /// Select modules to activate based on query analysis
    pub fn select_modules(&self, analysis: &QueryAnalysis) -> Vec<&PromptModule> {
        let mut activated: Vec<&PromptModule> = Vec::new();
        let mut excluded: HashSet<String> = HashSet::new();

        // First, collect all modules that should activate
        let mut candidates: Vec<(&PromptModule, bool)> = self
            .modules
            .values()
            .filter(|m| m.enabled)
            .map(|m| {
                let should_activate = if m.triggers.is_empty() {
                    // No triggers = check if it's a default module
                    self.config.default_modules.contains(&m.id)
                } else {
                    self.trigger_evaluator.evaluate_module_triggers(&m.triggers, analysis)
                };
                (m, should_activate)
            })
            .collect();

        // Sort by priority (higher first)
        candidates.sort_by(|a, b| b.0.priority.cmp(&a.0.priority));

        // Process candidates, respecting exclusions and dependencies
        for (module, should_activate) in candidates {
            if !should_activate {
                continue;
            }

            // Check if excluded by an already-activated module
            if excluded.contains(&module.id) {
                debug!("Module {} excluded by mutual exclusion", module.id);
                continue;
            }

            // Check dependencies
            let deps_satisfied = module.dependencies.iter().all(|dep| {
                activated.iter().any(|m| &m.id == dep) || self.config.default_modules.contains(dep)
            });

            if !deps_satisfied {
                debug!("Module {} skipped: dependencies not satisfied", module.id);
                continue;
            }

            // Activate this module
            activated.push(module);

            // Add its exclusions
            for ex in &module.mutual_exclusions {
                excluded.insert(ex.clone());
            }
        }

        debug!("Selected {} modules for activation", activated.len());
        activated
    }

    /// Compose a prompt from analysis and context
    pub fn compose(
        &self,
        analysis: &QueryAnalysis,
        context_objects: &[ContextObject],
        conversation_summary: Option<&str>,
    ) -> Result<ComposedPrompt> {
        let start = Instant::now();

        // Select modules to activate
        let activated_modules = self.select_modules(analysis);

        // Build template context
        let prompt_context = self.build_prompt_context(
            analysis,
            context_objects,
            conversation_summary,
        );

        // Separate modules by section
        let mut system_parts: Vec<String> = Vec::new();
        let mut context_parts: Vec<String> = Vec::new();
        let mut suffix_parts: Vec<String> = Vec::new();

        // Always include base system prompt
        if !self.config.base_system_prompt.is_empty() {
            system_parts.push(self.config.base_system_prompt.clone());
        }

        // Process activated modules
        for module in &activated_modules {
            let rendered = if TemplateEngine::is_template(&module.content) {
                self.template_engine
                    .render(&module.content, &prompt_context)
                    .map_err(|e| DpaError::TemplateError(e))?
            } else {
                module.content.clone()
            };

            match module.section {
                PromptSection::System => system_parts.push(rendered),
                PromptSection::Context => context_parts.push(rendered),
                PromptSection::Suffix => suffix_parts.push(rendered),
                _ => {} // Examples and Query are handled separately
            }
        }

        // Build context from context objects
        if !context_objects.is_empty() {
            let context_str = self.format_context_objects(context_objects);
            context_parts.push(context_str);
        }

        // Select examples
        let selected_examples = self.select_examples(analysis);
        let examples_count = selected_examples.len();

        // Estimate tokens (rough approximation: 4 chars per token)
        let system_str = system_parts.join("\n\n");
        let context_str = context_parts.join("\n\n");
        let suffix_str = suffix_parts.join("\n\n");
        let examples_tokens: usize = selected_examples
            .iter()
            .map(|e| (e.user.len() + e.assistant.len()) / 4)
            .sum();

        let estimated_tokens = system_str.len() / 4
            + context_str.len() / 4
            + analysis.original_query.len() / 4
            + suffix_str.len() / 4
            + examples_tokens;

        let composition_time = start.elapsed().as_millis() as u64;

        Ok(ComposedPrompt {
            system: system_str,
            context: context_str,
            examples: selected_examples,
            query: analysis.original_query.clone(),
            suffix: suffix_str,
            estimated_tokens,
            activated_modules: activated_modules.iter().map(|m| m.id.clone()).collect(),
            composition_metadata: CompositionMetadata {
                composition_time_ms: composition_time,
                modules_evaluated: self.modules.len(),
                modules_activated: activated_modules.len(),
                context_objects_included: context_objects.len(),
                examples_included: examples_count,
                tokens_used: estimated_tokens,
                token_budget: self.config.module_token_budget
                    + self.config.context_token_budget
                    + self.config.example_token_budget,
            },
        })
    }

    /// Build the prompt context for template rendering
    fn build_prompt_context(
        &self,
        analysis: &QueryAnalysis,
        context_objects: &[ContextObject],
        conversation_summary: Option<&str>,
    ) -> PromptContext {
        PromptContext {
            query: analysis.original_query.clone(),
            analysis: AnalysisContext {
                intent: analysis.primary_intent.to_string(),
                sentiment_score: analysis.sentiment.score,
                sentiment_label: format!("{:?}", analysis.sentiment.label),
                keywords: analysis.keywords.iter().map(|k| k.text.clone()).collect(),
                topics: analysis.topics.clone(),
                complexity: analysis.complexity,
                requires_knowledge: analysis.requires_knowledge,
                requires_reasoning: analysis.requires_reasoning,
            },
            context_objects: context_objects
                .iter()
                .map(|co| ContextObjectContext {
                    id: co.id.to_string(),
                    content: co.content.clone(),
                    object_type: format!("{:?}", co.object_type),
                    topics: co.topics.clone(),
                    summary: co.summary.clone(),
                })
                .collect(),
            conversation_summary: conversation_summary.map(String::from),
            current_datetime: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        }
    }

    /// Format context objects for inclusion in prompt
    fn format_context_objects(&self, objects: &[ContextObject]) -> String {
        let mut parts = Vec::new();

        for obj in objects {
            let content = obj.summary.as_ref().unwrap_or(&obj.content);
            parts.push(format!(
                "<context type=\"{:?}\" topics=\"{}\">\n{}\n</context>",
                obj.object_type,
                obj.topics.join(", "),
                content
            ));
        }

        parts.join("\n\n")
    }

    /// Select relevant examples for few-shot learning
    fn select_examples(&self, analysis: &QueryAnalysis) -> Vec<Example> {
        if self.examples.is_empty() {
            return Vec::new();
        }

        // Filter examples by topic overlap
        let mut scored_examples: Vec<(usize, &Example)> = self
            .examples
            .iter()
            .enumerate()
            .map(|(i, ex)| {
                let topic_overlap = ex
                    .topics
                    .iter()
                    .filter(|t| analysis.topics.contains(t))
                    .count();
                (topic_overlap, ex)
            })
            .collect();

        // Sort by topic overlap (descending)
        scored_examples.sort_by(|a, b| b.0.cmp(&a.0));

        // Take top examples up to limit
        scored_examples
            .into_iter()
            .take(self.config.max_examples)
            .map(|(_, ex)| ex.clone())
            .collect()
    }

    /// Get a module by ID
    pub fn get_module(&self, id: &str) -> Option<&PromptModule> {
        self.modules.get(id)
    }

    /// List all registered module IDs
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }
}

/// File format for loading modules
#[derive(Debug, serde::Deserialize)]
struct ModulesFile {
    #[serde(default)]
    modules: Vec<PromptModule>,
}

impl std::fmt::Debug for PromptBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptBuilder")
            .field("modules_count", &self.modules.len())
            .field("examples_count", &self.examples.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Create default prompt modules for common use cases
pub fn create_default_modules() -> Vec<PromptModule> {
    vec![
        PromptModule {
            id: "coding_mode".to_string(),
            name: "Coding Mode".to_string(),
            content: r#"You are an expert software engineer. When writing code:
- Prefer clarity over cleverness
- Include appropriate error handling
- Follow language idioms and best practices
- Explain non-obvious design decisions"#.to_string(),
            description: "Activates for code-related queries".to_string(),
            priority: 100,
            triggers: vec![
                super::triggers::Trigger::KeywordPresent {
                    keywords: vec![
                        "code".to_string(), "function".to_string(), "implement".to_string(),
                        "bug".to_string(), "error".to_string(), "compile".to_string(),
                        "program".to_string(), "class".to_string(), "method".to_string(),
                    ],
                    case_sensitive: false,
                },
                super::triggers::Trigger::IntentMatch {
                    intents: vec!["command".to_string()],
                },
            ],
            section: PromptSection::System,
            ..Default::default()
        },
        PromptModule {
            id: "analysis_mode".to_string(),
            name: "Analysis Mode".to_string(),
            content: r#"You are a data analyst. When analyzing:
- Consider multiple perspectives
- Support conclusions with evidence
- Identify potential biases or limitations
- Present findings clearly and objectively"#.to_string(),
            description: "Activates for analysis-related queries".to_string(),
            priority: 90,
            triggers: vec![
                super::triggers::Trigger::KeywordPresent {
                    keywords: vec![
                        "analyze".to_string(), "analysis".to_string(), "data".to_string(),
                        "trend".to_string(), "pattern".to_string(), "statistics".to_string(),
                    ],
                    case_sensitive: false,
                },
            ],
            section: PromptSection::System,
            ..Default::default()
        },
        PromptModule {
            id: "memory_recall".to_string(),
            name: "Memory Recall".to_string(),
            content: r#"The user is referencing previous context. Pay special attention to:
{{#each context_objects}}
- {{this.content}}
{{/each}}"#.to_string(),
            description: "Activates when memory triggers are detected".to_string(),
            priority: 80,
            triggers: vec![
                super::triggers::Trigger::MemoryTriggerDetected {
                    min_confidence: 0.5,
                },
            ],
            section: PromptSection::Context,
            ..Default::default()
        },
        PromptModule {
            id: "clarification_mode".to_string(),
            name: "Clarification Mode".to_string(),
            content: r#"The user is asking for clarification. Be sure to:
- Reference the specific point they're asking about
- Provide more detail than before
- Use different wording or examples
- Check if they have follow-up questions"#.to_string(),
            description: "Activates for clarification requests".to_string(),
            priority: 70,
            triggers: vec![
                super::triggers::Trigger::IntentMatch {
                    intents: vec!["clarification".to_string()],
                },
            ],
            section: PromptSection::System,
            ..Default::default()
        },
        PromptModule {
            id: "complex_reasoning".to_string(),
            name: "Complex Reasoning".to_string(),
            content: r#"This query requires careful reasoning. Approach it step by step:
1. Identify the core problem
2. Break it into manageable parts
3. Address each part systematically
4. Synthesize into a complete answer"#.to_string(),
            description: "Activates for complex reasoning tasks".to_string(),
            priority: 60,
            triggers: vec![
                super::triggers::Trigger::RequiresReasoning,
                super::triggers::Trigger::ComplexityRange {
                    min: 0.7,
                    max: 1.0,
                },
            ],
            section: PromptSection::System,
            ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::types::*;

    fn mock_analysis() -> QueryAnalysis {
        QueryAnalysis {
            original_query: "How do I implement a sorting algorithm in Rust?".to_string(),
            sentiment: Sentiment::default(),
            primary_intent: Intent::Command,
            topics: vec!["programming".to_string(), "rust".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_module_selection() {
        let config = ModuleConfig::default();
        let mut builder = PromptBuilder::new(config);

        builder.register_modules(create_default_modules());

        let analysis = mock_analysis();
        let selected = builder.select_modules(&analysis);

        // Should activate coding_mode due to "implement" keyword and Command intent
        assert!(selected.iter().any(|m| m.id == "coding_mode"));
    }

    #[test]
    fn test_compose_prompt() {
        let config = ModuleConfig {
            base_system_prompt: "You are a helpful assistant.".to_string(),
            ..Default::default()
        };
        let mut builder = PromptBuilder::new(config);
        builder.register_modules(create_default_modules());

        let analysis = mock_analysis();
        let prompt = builder.compose(&analysis, &[], None).unwrap();

        assert!(!prompt.system.is_empty());
        assert_eq!(prompt.query, analysis.original_query);
        assert!(prompt.activated_modules.contains(&"coding_mode".to_string()));
    }
}
