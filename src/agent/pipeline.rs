//! Main agent pipeline

use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument, warn};

use super::state::ConversationState;
use super::types::{AgentConfig, AgentMetrics, AgentResponse, StateSnapshot};
use crate::analyzer::QueryAnalyzer;
use crate::context::{ContextEngine, ContextEngineConfig};
use crate::embeddings::{create_embedding_provider, EmbeddingProvider};
use crate::llm::{create_provider, CompletionOptions, LlmProvider};
use crate::prompt::{create_default_modules, ModuleConfig, PromptBuilder};
use crate::{DpaError, Result};

/// The main Dynamic Prompt Agent
pub struct Agent {
    /// Small LLM for analysis
    small_llm: Arc<dyn LlmProvider>,

    /// Large LLM for generation
    large_llm: Arc<dyn LlmProvider>,

    /// Query analyzer
    analyzer: QueryAnalyzer,

    /// Prompt builder
    prompt_builder: PromptBuilder,

    /// Context engine
    context_engine: ContextEngine,

    /// Conversation state
    state: ConversationState,

    /// Configuration
    config: AgentConfig,
}

impl Agent {
    /// Create a new agent with the given configuration
    pub async fn new(config: AgentConfig) -> Result<Self> {
        // Create LLM providers - convert Box to Arc
        let small_llm: Arc<dyn LlmProvider> = Arc::from(create_provider(&config.small_llm).await?);
        let large_llm: Arc<dyn LlmProvider> = Arc::from(create_provider(&config.large_llm).await?);

        // Create embedding provider
        let embedding_provider: Arc<dyn EmbeddingProvider> =
            create_embedding_provider(config.embedding_dim)?;

        // Create analyzer
        let analyzer = QueryAnalyzer::new(Arc::clone(&small_llm));

        // Create prompt builder
        let mut prompt_builder = PromptBuilder::new(config.prompt_config.clone());
        prompt_builder.register_modules(create_default_modules());

        // Create context engine
        let context_engine = if let Some(ref path) = config.persistence_path {
            ContextEngine::with_persistence(
                Arc::clone(&small_llm),
                embedding_provider,
                config.context_config.clone(),
                path,
            )?
        } else {
            ContextEngine::new(
                Arc::clone(&small_llm),
                embedding_provider,
                config.context_config.clone(),
            )
        };

        // Create conversation state
        let session_id = config
            .session_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let state = ConversationState::new(&session_id);

        Ok(Self {
            small_llm: small_llm.into(),
            large_llm: large_llm.into(),
            analyzer,
            prompt_builder,
            context_engine,
            state,
            config,
        })
    }

    /// Process a user message and generate a response
    #[instrument(skip(self), fields(message_len = user_message.len()))]
    pub async fn process(&mut self, user_message: &str) -> Result<AgentResponse> {
        let start = Instant::now();
        let mut metrics = AgentMetrics::default();

        // Step 1: Analyze the query
        let analysis_start = Instant::now();
        let context_summary = self.state.conversation_summary.as_deref();
        let analysis = self.analyzer.analyze(user_message, context_summary).await?;
        metrics.analysis_time_ms = analysis_start.elapsed().as_millis() as u64;

        if let Some(latency) = analysis.latency_ms {
            debug!("Query analysis completed in {}ms", latency);
        }

        // Step 2: Retrieve relevant context
        let retrieval_start = Instant::now();
        let mut context_objects = self.context_engine.retrieve(&analysis).await?;

        // Also retrieve by memory triggers if any
        if !analysis.memory_triggers.is_empty() {
            let trigger_context = self.context_engine.retrieve_by_triggers(&analysis).await?;
            for obj in trigger_context {
                if !context_objects.iter().any(|o| o.id == obj.id) {
                    context_objects.push(obj);
                }
            }
        }
        metrics.retrieval_time_ms = retrieval_start.elapsed().as_millis() as u64;
        metrics.context_objects_retrieved = context_objects.len();

        debug!("Retrieved {} context objects", context_objects.len());

        // Step 3: Compose the prompt
        let composition_start = Instant::now();
        let composed_prompt = self.prompt_builder.compose(
            &analysis,
            &context_objects,
            self.state.conversation_summary.as_deref(),
        )?;
        metrics.composition_time_ms = composition_start.elapsed().as_millis() as u64;
        metrics.modules_activated = composed_prompt.activated_modules.len();

        debug!(
            "Composed prompt with {} modules, {} estimated tokens",
            composed_prompt.activated_modules.len(),
            composed_prompt.estimated_tokens
        );

        // Step 4: Generate response with large LLM
        let generation_start = Instant::now();
        let options = CompletionOptions {
            system: Some(composed_prompt.system.clone()),
            max_tokens: Some(self.config.large_llm.max_tokens),
            temperature: Some(self.config.large_llm.temperature),
            ..Default::default()
        };

        let response = self
            .large_llm
            .complete(&composed_prompt.user_message(), &options)
            .await?;

        metrics.generation_time_ms = generation_start.elapsed().as_millis() as u64;
        metrics.large_llm_input_tokens = response.input_tokens;
        metrics.large_llm_output_tokens = response.output_tokens;

        let content = response.content;

        // Step 5: Update state and context
        self.state.add_exchange(user_message, &content);
        self.context_engine.next_turn();

        // Ingest the conversation turn into context store
        if let Err(e) = self.context_engine.ingest_turn(user_message, &content).await {
            warn!("Failed to ingest conversation turn: {}", e);
        }

        // Calculate total time
        metrics.total_time_ms = start.elapsed().as_millis() as u64;

        // Create state snapshot
        let summary = self.context_engine.get_context_summary()?;
        let state_snapshot = StateSnapshot {
            turn_number: self.state.turn_number,
            context_objects_count: summary.total_objects,
            context_tokens: summary.total_tokens,
            session_id: Some(self.state.session_id.clone()),
        };

        info!(
            "Request completed in {}ms (analysis: {}ms, retrieval: {}ms, composition: {}ms, generation: {}ms)",
            metrics.total_time_ms,
            metrics.analysis_time_ms,
            metrics.retrieval_time_ms,
            metrics.composition_time_ms,
            metrics.generation_time_ms
        );

        Ok(AgentResponse {
            content,
            analysis,
            composed_prompt,
            metrics,
            state_snapshot,
        })
    }

    /// Process a message without updating state (for testing/debugging)
    pub async fn process_stateless(&self, user_message: &str) -> Result<String> {
        // Analyze
        let analysis = self.analyzer.analyze(user_message, None).await?;

        // Retrieve context
        let context_objects = self.context_engine.retrieve(&analysis).await?;

        // Compose prompt
        let composed_prompt = self.prompt_builder.compose(&analysis, &context_objects, None)?;

        // Generate
        let options = CompletionOptions {
            system: Some(composed_prompt.system.clone()),
            max_tokens: Some(self.config.large_llm.max_tokens),
            temperature: Some(self.config.large_llm.temperature),
            ..Default::default()
        };

        let response = self
            .large_llm
            .complete(&composed_prompt.user_message(), &options)
            .await?;

        Ok(response.content)
    }

    /// Add context directly to the context engine
    pub async fn add_context(&self, content: &str, source: &str) -> Result<Vec<uuid::Uuid>> {
        self.context_engine.ingest(content, source).await
    }

    /// Get the current conversation state
    pub fn state(&self) -> &ConversationState {
        &self.state
    }

    /// Get mutable conversation state
    pub fn state_mut(&mut self) -> &mut ConversationState {
        &mut self.state
    }

    /// Reset the conversation
    pub fn reset(&mut self) -> Result<()> {
        self.state.reset();
        self.context_engine.clear()?;
        Ok(())
    }

    /// Get context summary
    pub fn context_summary(&self) -> Result<crate::context::ContextSummary> {
        self.context_engine.get_context_summary()
    }

    /// Flush any pending writes to disk
    pub fn flush(&self) -> Result<()> {
        self.context_engine.flush()
    }

    /// Update the base system prompt
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.config.prompt_config.base_system_prompt = prompt.to_string();
        self.prompt_builder = PromptBuilder::new(self.config.prompt_config.clone());
        self.prompt_builder.register_modules(create_default_modules());
    }

    /// Register additional prompt modules
    pub fn register_modules(&mut self, modules: Vec<crate::prompt::PromptModule>) {
        self.prompt_builder.register_modules(modules);
    }

    /// Get metrics from the last request (if any stored)
    pub fn get_model_info(&self) -> (crate::llm::ModelInfo, crate::llm::ModelInfo) {
        (self.small_llm.model_info(), self.large_llm.model_info())
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("session_id", &self.state.session_id)
            .field("turn_number", &self.state.turn_number)
            .field("config", &self.config)
            .finish()
    }
}

/// Builder for creating an agent with a fluent API
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
        }
    }

    pub fn small_llm(mut self, provider: &str, model: &str) -> Self {
        self.config.small_llm.provider = provider.to_string();
        self.config.small_llm.model = model.to_string();
        self
    }

    pub fn large_llm(mut self, provider: &str, model: &str) -> Self {
        self.config.large_llm.provider = provider.to_string();
        self.config.large_llm.model = model.to_string();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.config.small_llm.api_key = Some(key.to_string());
        self.config.large_llm.api_key = Some(key.to_string());
        self
    }

    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.config.prompt_config.base_system_prompt = prompt.to_string();
        self
    }

    pub fn session_id(mut self, id: &str) -> Self {
        self.config.session_id = Some(id.to_string());
        self
    }

    pub fn persistence(mut self, path: &str) -> Self {
        self.config.persistence_path = Some(path.to_string());
        self
    }

    pub fn context_budget(mut self, tokens: usize) -> Self {
        self.config.context_config.token_budget = tokens;
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.config.context_config.top_k = k;
        self
    }

    pub fn debug(mut self, enabled: bool) -> Self {
        self.config.debug = enabled;
        self
    }

    pub async fn build(self) -> Result<Agent> {
        Agent::new(self.config).await
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
