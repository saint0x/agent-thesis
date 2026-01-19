//! Conversation state management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Maximum number of recent exchanges to keep in memory
const MAX_RECENT_EXCHANGES: usize = 10;

/// A single exchange (user message + assistant response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange {
    /// User's message
    pub user_message: String,

    /// Assistant's response
    pub assistant_response: String,

    /// Turn number
    pub turn_number: usize,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Summary (if compressed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Conversation state for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    /// Session identifier
    pub session_id: String,

    /// Current turn number
    pub turn_number: usize,

    /// Recent exchanges (kept in memory for quick access)
    pub recent_exchanges: VecDeque<Exchange>,

    /// Running summary of the conversation
    pub conversation_summary: Option<String>,

    /// When the session started
    pub started_at: DateTime<Utc>,

    /// When the session was last active
    pub last_active_at: DateTime<Utc>,

    /// Custom metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl ConversationState {
    /// Create a new conversation state
    pub fn new(session_id: &str) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.to_string(),
            turn_number: 0,
            recent_exchanges: VecDeque::new(),
            conversation_summary: None,
            started_at: now,
            last_active_at: now,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add an exchange to the conversation
    pub fn add_exchange(&mut self, user_message: &str, assistant_response: &str) {
        self.turn_number += 1;
        self.last_active_at = Utc::now();

        let exchange = Exchange {
            user_message: user_message.to_string(),
            assistant_response: assistant_response.to_string(),
            turn_number: self.turn_number,
            timestamp: self.last_active_at,
            summary: None,
        };

        self.recent_exchanges.push_back(exchange);

        // Keep only recent exchanges
        while self.recent_exchanges.len() > MAX_RECENT_EXCHANGES {
            self.recent_exchanges.pop_front();
        }
    }

    /// Get the last N exchanges
    pub fn get_recent(&self, n: usize) -> Vec<&Exchange> {
        self.recent_exchanges
            .iter()
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Get formatted conversation history
    pub fn get_formatted_history(&self, max_turns: usize) -> String {
        let exchanges = self.get_recent(max_turns);

        exchanges
            .iter()
            .map(|e| format!("User: {}\n\nAssistant: {}", e.user_message, e.assistant_response))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    /// Update the conversation summary
    pub fn set_summary(&mut self, summary: &str) {
        self.conversation_summary = Some(summary.to_string());
    }

    /// Get conversation context (summary + recent history)
    pub fn get_context(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref summary) = self.conversation_summary {
            parts.push(format!("Conversation summary:\n{}", summary));
        }

        let recent = self.get_formatted_history(3);
        if !recent.is_empty() {
            parts.push(format!("Recent exchanges:\n{}", recent));
        }

        parts.join("\n\n")
    }

    /// Get session duration
    pub fn duration(&self) -> chrono::Duration {
        self.last_active_at - self.started_at
    }

    /// Check if session is stale (inactive for given duration)
    pub fn is_stale(&self, max_inactive_hours: i64) -> bool {
        let inactive = Utc::now() - self.last_active_at;
        inactive.num_hours() > max_inactive_hours
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Reset the conversation
    pub fn reset(&mut self) {
        self.turn_number = 0;
        self.recent_exchanges.clear();
        self.conversation_summary = None;
        self.started_at = Utc::now();
        self.last_active_at = self.started_at;
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new(&uuid::Uuid::new_v4().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_exchange() {
        let mut state = ConversationState::new("test-session");

        state.add_exchange("Hello", "Hi there!");
        assert_eq!(state.turn_number, 1);
        assert_eq!(state.recent_exchanges.len(), 1);

        state.add_exchange("How are you?", "I'm doing well!");
        assert_eq!(state.turn_number, 2);
        assert_eq!(state.recent_exchanges.len(), 2);
    }

    #[test]
    fn test_get_recent() {
        let mut state = ConversationState::new("test-session");

        for i in 0..5 {
            state.add_exchange(&format!("Message {}", i), &format!("Response {}", i));
        }

        let recent = state.get_recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].turn_number, 3);
        assert_eq!(recent[2].turn_number, 5);
    }

    #[test]
    fn test_max_exchanges() {
        let mut state = ConversationState::new("test-session");

        for i in 0..15 {
            state.add_exchange(&format!("Message {}", i), &format!("Response {}", i));
        }

        assert_eq!(state.recent_exchanges.len(), MAX_RECENT_EXCHANGES);
        assert_eq!(state.turn_number, 15);
    }
}
