//! OpenAI LLM provider implementation

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::{CompletionOptions, CompletionResponse, LlmConfig, LlmProvider, ModelInfo};
use crate::{DpaError, Result};

const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";
const OPENAI_EMBED_URL: &str = "https://api.openai.com/v1/embeddings";

/// OpenAI provider
#[derive(Debug)]
pub struct OpenAIProvider {
    config: LlmConfig,
    client: Client,
    api_key: String,
}

impl OpenAIProvider {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| DpaError::ConfigError("OPENAI_API_KEY not set".to_string()))?;

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
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAIError {
    error: OpenAIErrorDetail,
}

#[derive(Deserialize)]
struct OpenAIErrorDetail {
    message: String,
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn complete(&self, prompt: &str, options: &CompletionOptions) -> Result<CompletionResponse> {
        let start = Instant::now();

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or(OPENAI_CHAT_URL);

        let mut messages = Vec::new();

        if let Some(system) = &options.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request = OpenAIChatRequest {
            model: self.config.model.clone(),
            messages,
            max_tokens: Some(options.max_tokens.unwrap_or(self.config.max_tokens)),
            temperature: Some(options.temperature.unwrap_or(self.config.temperature)),
            response_format: if options.json_mode {
                Some(ResponseFormat {
                    format_type: "json_object".to_string(),
                })
            } else {
                None
            },
            stop: options.stop_sequences.clone(),
        };

        let response = self
            .client
            .post(base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            let error: OpenAIError = serde_json::from_str(&body)
                .map_err(|_| DpaError::LlmError(format!("HTTP {}: {}", status, body)))?;
            return Err(DpaError::LlmError(error.error.message));
        }

        let response: OpenAIChatResponse = serde_json::from_str(&body)?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let content = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let stop_reason = response.choices.first().and_then(|c| c.finish_reason.clone());

        Ok(CompletionResponse {
            content,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            model: response.model,
            stop_reason,
            latency_ms,
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIEmbedRequest {
            model: "text-embedding-3-small".to_string(),
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post(OPENAI_EMBED_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            let error: OpenAIError = serde_json::from_str(&body)
                .map_err(|_| DpaError::EmbeddingError(format!("HTTP {}: {}", status, body)))?;
            return Err(DpaError::EmbeddingError(error.error.message));
        }

        let response: OpenAIEmbedResponse = serde_json::from_str(&body)?;
        Ok(response.data.into_iter().map(|e| e.embedding).collect())
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn model_info(&self) -> ModelInfo {
        let context_window = match self.config.model.as_str() {
            "gpt-4o" | "gpt-4o-mini" => 128000,
            "gpt-4-turbo" => 128000,
            "gpt-4" => 8192,
            "gpt-3.5-turbo" => 16385,
            _ => 8192,
        };

        ModelInfo {
            provider: "openai".to_string(),
            model: self.config.model.clone(),
            context_window,
            supports_json_mode: true,
            supports_streaming: true,
        }
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}
