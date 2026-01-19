//! Anthropic (Claude) LLM provider implementation

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::{CompletionOptions, CompletionResponse, LlmConfig, LlmProvider, ModelInfo};
use crate::{DpaError, Result};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Claude provider
#[derive(Debug)]
pub struct AnthropicProvider {
    config: LlmConfig,
    client: Client,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| DpaError::ConfigError("ANTHROPIC_API_KEY not set".to_string()))?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| DpaError::LlmError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            api_key,
        })
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
    #[serde(rename = "type")]
    _type: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    message: String,
    #[serde(rename = "type")]
    _type: String,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, prompt: &str, options: &CompletionOptions) -> Result<CompletionResponse> {
        let start = Instant::now();

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or(ANTHROPIC_API_URL);

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(self.config.max_tokens),
            system: options.system.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: Some(options.temperature.unwrap_or(self.config.temperature)),
            stop_sequences: options.stop_sequences.clone(),
        };

        let response = self
            .client
            .post(base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            let error: AnthropicError = serde_json::from_str(&body)
                .map_err(|_| DpaError::LlmError(format!("HTTP {}: {}", status, body)))?;
            return Err(DpaError::LlmError(error.error.message));
        }

        let response: AnthropicResponse = serde_json::from_str(&body)?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let content = response
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse {
            content,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            model: response.model,
            stop_reason: response.stop_reason,
            latency_ms,
        })
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(DpaError::EmbeddingError(
            "Anthropic does not support embeddings".to_string(),
        ))
    }

    fn supports_embeddings(&self) -> bool {
        false
    }

    fn model_info(&self) -> ModelInfo {
        let context_window = match self.config.model.as_str() {
            m if m.contains("opus") => 200000,
            m if m.contains("sonnet") => 200000,
            m if m.contains("haiku") => 200000,
            _ => 100000,
        };

        ModelInfo {
            provider: "anthropic".to_string(),
            model: self.config.model.clone(),
            context_window,
            supports_json_mode: false,
            supports_streaming: true,
        }
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}
