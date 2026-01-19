//! Dynamic Prompt Builder Module
//!
//! This module provides:
//! - Prompt modules with conditional triggers
//! - Dynamic prompt composition based on query analysis
//! - Template rendering with Handlebars
//! - Few-shot example selection

mod types;
mod triggers;
mod composer;
mod templates;

pub use types::{PromptModule, ComposedPrompt, ModuleConfig, PromptSection};
pub use triggers::{Trigger, TriggerEvaluator};
pub use composer::{PromptBuilder, create_default_modules};
pub use templates::TemplateEngine;
