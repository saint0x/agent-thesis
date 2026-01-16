//! Template engine for prompt rendering

use handlebars::{Handlebars, RenderError};
use serde::Serialize;
use std::collections::HashMap;

/// Template engine wrapper around Handlebars
pub struct TemplateEngine {
    handlebars: Handlebars<'static>,
}

impl TemplateEngine {
    /// Create a new template engine
    pub fn new() -> Self {
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(false);

        // Register custom helpers
        Self::register_helpers(&mut handlebars);

        Self { handlebars }
    }

    /// Register custom Handlebars helpers
    fn register_helpers(handlebars: &mut Handlebars<'static>) {
        // Helper to join array elements
        handlebars.register_helper(
            "join",
            Box::new(|h: &handlebars::Helper,
                      _: &Handlebars,
                      _: &handlebars::Context,
                      _: &mut handlebars::RenderContext,
                      out: &mut dyn handlebars::Output|
             -> handlebars::HelperResult {
                let param = h.param(0).and_then(|v| v.value().as_array());
                let sep = h.param(1).and_then(|v| v.value().as_str()).unwrap_or(", ");

                if let Some(arr) = param {
                    let joined: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    out.write(&joined.join(sep))?;
                }

                Ok(())
            }),
        );

        // Helper to truncate text
        handlebars.register_helper(
            "truncate",
            Box::new(|h: &handlebars::Helper,
                      _: &Handlebars,
                      _: &handlebars::Context,
                      _: &mut handlebars::RenderContext,
                      out: &mut dyn handlebars::Output|
             -> handlebars::HelperResult {
                let text = h.param(0).and_then(|v| v.value().as_str()).unwrap_or("");
                let max_len: usize = h
                    .param(1)
                    .and_then(|v| v.value().as_u64())
                    .unwrap_or(100) as usize;

                if text.len() > max_len {
                    out.write(&text[..max_len])?;
                    out.write("...")?;
                } else {
                    out.write(text)?;
                }

                Ok(())
            }),
        );

        // Helper to check if value is empty
        handlebars.register_helper(
            "is_empty",
            Box::new(|h: &handlebars::Helper,
                      _: &Handlebars,
                      _: &handlebars::Context,
                      _: &mut handlebars::RenderContext,
                      out: &mut dyn handlebars::Output|
             -> handlebars::HelperResult {
                let is_empty = h.param(0).map_or(true, |v| {
                    let val = v.value();
                    val.is_null()
                        || val.as_str().map_or(false, |s| s.is_empty())
                        || val.as_array().map_or(false, |a| a.is_empty())
                });

                out.write(if is_empty { "true" } else { "false" })?;
                Ok(())
            }),
        );
    }

    /// Register a named template
    pub fn register_template(&mut self, name: &str, template: &str) -> Result<(), RenderError> {
        self.handlebars
            .register_template_string(name, template)
            .map_err(|e| RenderError::from(e))
    }

    /// Render a template string with context
    pub fn render<T: Serialize>(&self, template: &str, context: &T) -> Result<String, RenderError> {
        self.handlebars.render_template(template, context)
    }

    /// Render a named template with context
    pub fn render_named<T: Serialize>(
        &self,
        name: &str,
        context: &T,
    ) -> Result<String, RenderError> {
        self.handlebars.render(name, context)
    }

    /// Check if a template contains any Handlebars expressions
    pub fn is_template(text: &str) -> bool {
        text.contains("{{") && text.contains("}}")
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for rendering prompt templates
#[derive(Debug, Clone, Serialize)]
pub struct PromptContext {
    /// User's query
    pub query: String,

    /// Analysis results
    pub analysis: AnalysisContext,

    /// Retrieved context objects
    pub context_objects: Vec<ContextObjectContext>,

    /// Conversation history summary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_summary: Option<String>,

    /// Current date/time
    pub current_datetime: String,

    /// Custom variables
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AnalysisContext {
    pub intent: String,
    pub sentiment_score: f32,
    pub sentiment_label: String,
    pub keywords: Vec<String>,
    pub topics: Vec<String>,
    pub complexity: f32,
    pub requires_knowledge: bool,
    pub requires_reasoning: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextObjectContext {
    pub id: String,
    pub content: String,
    pub object_type: String,
    pub topics: Vec<String>,
    pub summary: Option<String>,
}

impl Default for PromptContext {
    fn default() -> Self {
        Self {
            query: String::new(),
            analysis: AnalysisContext::default(),
            context_objects: Vec::new(),
            conversation_summary: None,
            current_datetime: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_template() {
        let engine = TemplateEngine::new();
        let context = serde_json::json!({
            "name": "World"
        });

        let result = engine.render("Hello, {{name}}!", &context).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_join_helper() {
        let engine = TemplateEngine::new();
        let context = serde_json::json!({
            "items": ["a", "b", "c"]
        });

        let result = engine.render("Items: {{join items \", \"}}", &context).unwrap();
        assert_eq!(result, "Items: a, b, c");
    }

    #[test]
    fn test_conditional() {
        let engine = TemplateEngine::new();
        let context = serde_json::json!({
            "show": true,
            "message": "Hello"
        });

        let result = engine
            .render("{{#if show}}{{message}}{{/if}}", &context)
            .unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_is_template() {
        assert!(TemplateEngine::is_template("Hello {{name}}"));
        assert!(!TemplateEngine::is_template("Hello world"));
    }
}
