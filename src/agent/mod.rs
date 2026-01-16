//! Agent Orchestration Module
//!
//! This module provides the main Agent that ties together:
//! - Query analysis
//! - Dynamic prompt building
//! - Modular context management
//! - LLM interaction

mod types;
mod pipeline;
mod state;

pub use types::{AgentConfig, AgentResponse, AgentMetrics};
pub use pipeline::Agent;
pub use state::ConversationState;
