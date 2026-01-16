# Dynamic Prompt Agent Architecture

## A Thesis on Runtime-Adaptive Prompt Composition and Modular Context Management

---

## Executive Summary

This thesis proposes a novel agent architecture combining two complementary innovations:

1. **Dynamic Prompt Builder (DPB)**: A small, fast LLM performs real-time query analysis to dynamically compose prompts at runtime based on detected patterns, sentiment, keywords, and context requirements.

2. **Modular Context Engine (MCE)**: The same lightweight LLM partitions accumulated context into discrete, addressable memory objects that can be selectively retrieved and composed into the active context window.

Together, these systems create a fully modular, fully dynamic agent where both the prompt and the context window adapt intelligently to each interaction.

---

## State of the Art Analysis

### What Exists Today

| Approach | Strengths | Limitations |
|----------|-----------|-------------|
| **RAG (Retrieval-Augmented Generation)** | Scales beyond context limits, cost-effective | Lossy retrieval, semantic drift, chunk boundary issues |
| **Long Context Windows** (200K-1M tokens) | Full context preservation | Expensive, context rot, attention dilution |
| **KV Cache Compression** (KVzip, etc.) | 3-4x compression, maintains accuracy | Model-specific, doesn't address prompt composition |
| **Static Prompt Templates** | Predictable, maintainable | Inflexible, doesn't adapt to query nuance |
| **LLM Routing** (small→large model cascade) | Cost-efficient | Binary routing, not compositional |

### Research Gaps This Thesis Addresses

1. **No unified system** combines dynamic prompt composition with modular context management
2. **Existing approaches are reactive** (retrieve what's needed) rather than **proactive** (anticipate and prepare)
3. **Lossless partitioning** of context into recomposable units is underexplored
4. **Small LLM as orchestrator** pattern exists for routing but not for prompt/context construction
5. **Rust implementations** are emerging but lack this architectural sophistication

### Key Inspirations from Literature

- **[Mem0](https://arxiv.org/pdf/2504.19413)**: Graph-based memory representations for relational structures
- **[KVzip](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html)**: Intelligent compression preserving essential context
- **[TTT-E2E](https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time/)**: Context as training data, constant inference latency
- **[MasRouter](https://arxiv.org/html/2601.04861)**: Confidence-aware routing in multi-agent systems
- **[APE (Automatic Prompt Engineer)](https://arxiv.org/html/2401.14423v4)**: Dynamic instruction generation and selection
- **[Rig Framework](https://rig.rs/)**: Modular LLM applications in Rust
- **[graph-flow](https://github.com/a-agmon/rs-graph-llm)**: Stateful AI agent orchestration in Rust

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUERY ANALYZER (Small LLM)                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Sentiment   │ │   Keyword    │ │   Intent     │ │   Memory     │   │
│  │  Detection   │ │  Extraction  │ │Classification│ │   Triggers   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│   DYNAMIC PROMPT BUILDER      │   │   MODULAR CONTEXT ENGINE          │
│                               │   │                                   │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   Base System Prompt    │  │   │  │    Context Object Store     │  │
│  └─────────────────────────┘  │   │  │  ┌─────┐ ┌─────┐ ┌─────┐   │  │
│  ┌─────────────────────────┐  │   │  │  │ CO₁ │ │ CO₂ │ │ CO₃ │   │  │
│  │  Conditional Modules    │  │   │  │  └─────┘ └─────┘ └─────┘   │  │
│  │  ├─ If keyword X → +M₁  │  │   │  │  ┌─────┐ ┌─────┐ ┌─────┐   │  │
│  │  ├─ If sentiment Y → +M₂│  │   │  │  │ CO₄ │ │ CO₅ │ │ ... │   │  │
│  │  └─ If memory Z → +M₃   │  │   │  │  └─────┘ └─────┘ └─────┘   │  │
│  └─────────────────────────┘  │   │  └─────────────────────────────┘  │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────────┐  │
│  │   Few-Shot Selector     │  │   │  │   Relevance Scorer          │  │
│  └─────────────────────────┘  │   │  │   (cosine + recency + refs) │  │
│  ┌─────────────────────────┐  │   │  └─────────────────────────────┘  │
│  │   Tone/Style Adapter    │  │   │  ┌─────────────────────────────┐  │
│  └─────────────────────────┘  │   │  │   Context Composer          │  │
└───────────────────────────────┘   │  │   (assemble selected COs)   │  │
                    │               │  └─────────────────────────────┘  │
                    │               └───────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPOSED PROMPT + CONTEXT                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ [System: Base + M₁ + M₃] [Context: CO₂ + CO₅] [Query] [Examples] │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PRIMARY LLM (Large Model)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESPONSE + CONTEXT UPDATE                             │
│                (MCE partitions new context into COs)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Context Objects (COs)

A Context Object is an atomic, addressable unit of context with metadata:

```rust
struct ContextObject {
    id: Uuid,
    content: String,
    embedding: Vec<f32>,

    // Metadata for retrieval scoring
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u32,

    // Relational metadata
    references: Vec<Uuid>,        // COs this one references
    referenced_by: Vec<Uuid>,     // COs that reference this one

    // Classification
    object_type: ContextType,     // Fact, Instruction, Example, Conversation, etc.
    topics: Vec<String>,
    importance_score: f32,        // 0.0 - 1.0

    // Compression metadata
    is_compressed: bool,
    original_token_count: usize,
    compressed_token_count: usize,
}

enum ContextType {
    Fact,           // Standalone factual information
    Instruction,    // User preferences, rules
    Example,        // Few-shot examples
    Conversation,   // Dialog history segment
    ToolResult,     // Output from tool calls
    SystemState,    // Application state
}
```

### 2. Prompt Modules (PMs)

Conditional prompt segments that can be composed:

```rust
struct PromptModule {
    id: String,
    content: String,

    // Activation conditions
    triggers: Vec<Trigger>,

    // Composition metadata
    priority: i32,              // Higher = placed earlier
    mutual_exclusions: Vec<String>,  // IDs of incompatible modules
    dependencies: Vec<String>,  // IDs of required modules
}

enum Trigger {
    KeywordPresent(Vec<String>),
    KeywordAbsent(Vec<String>),
    SentimentRange { min: f32, max: f32 },
    IntentMatch(String),
    ContextObjectExists(Uuid),
    TopicPresent(String),
    UserPreference(String),
    ConversationLength { min: usize, max: Option<usize> },
    Custom(Box<dyn Fn(&QueryAnalysis) -> bool>),
}
```

### 3. Query Analysis Output

```rust
struct QueryAnalysis {
    original_query: String,

    // Sentiment analysis
    sentiment_score: f32,        // -1.0 (negative) to 1.0 (positive)
    sentiment_magnitude: f32,    // 0.0 (neutral) to 1.0 (strong)

    // Intent classification
    primary_intent: Intent,
    secondary_intents: Vec<Intent>,
    confidence: f32,

    // Extracted entities
    keywords: Vec<Keyword>,
    entities: Vec<Entity>,

    // Memory triggers
    memory_queries: Vec<String>, // Queries to search context store

    // Suggested prompt modules
    suggested_modules: Vec<String>,

    // Context requirements
    required_context_types: Vec<ContextType>,
    estimated_context_tokens: usize,
}

enum Intent {
    Question,
    Command,
    Clarification,
    Correction,
    Continuation,
    NewTopic,
    MetaQuery,  // Query about the agent itself
}
```

---

## Implementation Checklist

### Phase 1: Foundation (Rust Core Infrastructure)

#### 1.1 Project Setup
- [ ] Initialize Rust workspace with cargo
- [ ] Set up module structure:
  ```
  src/
  ├── lib.rs
  ├── main.rs
  ├── analyzer/          # Query analysis module
  │   ├── mod.rs
  │   ├── sentiment.rs
  │   ├── intent.rs
  │   ├── keywords.rs
  │   └── triggers.rs
  ├── prompt/            # Dynamic prompt builder
  │   ├── mod.rs
  │   ├── modules.rs
  │   ├── composer.rs
  │   └── templates.rs
  ├── context/           # Modular context engine
  │   ├── mod.rs
  │   ├── objects.rs
  │   ├── store.rs
  │   ├── partitioner.rs
  │   └── scorer.rs
  ├── llm/               # LLM interface abstractions
  │   ├── mod.rs
  │   ├── small.rs       # Small/fast model interface
  │   ├── large.rs       # Primary model interface
  │   └── providers.rs   # OpenAI, Anthropic, local, etc.
  ├── embeddings/        # Vector embedding utilities
  │   ├── mod.rs
  │   └── similarity.rs
  └── agent/             # Main agent orchestration
      ├── mod.rs
      ├── pipeline.rs
      └── state.rs
  ```
- [ ] Configure dependencies in `Cargo.toml`:
  - `tokio` (async runtime)
  - `serde` / `serde_json` (serialization)
  - `uuid` (identifiers)
  - `chrono` (timestamps)
  - `reqwest` (HTTP client for LLM APIs)
  - `fastembed` or `candle` (local embeddings)
  - `hnsw` or `usearch` (vector similarity search)
  - `tracing` (observability)

#### 1.2 LLM Interface Layer
- [ ] Define `LlmProvider` trait:
  ```rust
  #[async_trait]
  trait LlmProvider {
      async fn complete(&self, prompt: &str, options: &CompletionOptions) -> Result<String>;
      async fn embed(&self, text: &str) -> Result<Vec<f32>>;
      fn model_info(&self) -> ModelInfo;
  }
  ```
- [ ] Implement for small model (e.g., Qwen2.5-3B, Phi-3-mini, or API: Claude Haiku)
- [ ] Implement for large model (e.g., Claude Sonnet/Opus, GPT-4)
- [ ] Add rate limiting and retry logic
- [ ] Implement streaming support

#### 1.3 Embedding System
- [ ] Integrate embedding model (local via `candle` or API)
- [ ] Build vector store abstraction
- [ ] Implement cosine similarity scoring
- [ ] Add batch embedding support

---

### Phase 2: Query Analyzer Module

#### 2.1 Core Analysis Pipeline
- [ ] Design structured output schema for small LLM
- [ ] Implement `QueryAnalyzer` struct:
  ```rust
  struct QueryAnalyzer {
      llm: Arc<dyn LlmProvider>,
      analysis_prompt: PromptTemplate,
  }

  impl QueryAnalyzer {
      async fn analyze(&self, query: &str) -> Result<QueryAnalysis>;
  }
  ```
- [ ] Create analysis prompt template with examples
- [ ] Parse structured JSON output from small LLM
- [ ] Add fallback for malformed responses

#### 2.2 Sentiment Analysis
- [ ] Define sentiment scoring (-1.0 to 1.0 scale)
- [ ] Implement sentiment magnitude detection
- [ ] Create test cases covering edge cases
- [ ] Benchmark accuracy against labeled dataset

#### 2.3 Intent Classification
- [ ] Define intent taxonomy (Question, Command, Clarification, etc.)
- [ ] Implement multi-intent detection with confidence scores
- [ ] Add intent hierarchy (primary vs secondary)
- [ ] Create training examples for each intent

#### 2.4 Keyword & Entity Extraction
- [ ] Extract keywords with importance weights
- [ ] Identify named entities (people, places, technical terms)
- [ ] Detect domain-specific terminology
- [ ] Link entities to context objects when possible

#### 2.5 Memory Trigger Detection
- [ ] Identify implicit references to past context
- [ ] Generate semantic search queries
- [ ] Detect explicit memory requests ("remember when...")
- [ ] Score trigger confidence

---

### Phase 3: Dynamic Prompt Builder

#### 3.1 Prompt Module System
- [ ] Design module storage format (TOML/YAML config files)
- [ ] Implement `PromptModule` struct with triggers
- [ ] Build module registry with hot-reloading
- [ ] Create standard module library:
  - [ ] Tone adjusters (formal, casual, technical)
  - [ ] Safety modules (content filtering triggers)
  - [ ] Domain modules (coding, writing, analysis)
  - [ ] Memory modules (when context objects are relevant)
  - [ ] Clarification modules (when query is ambiguous)

#### 3.2 Trigger Evaluation
- [ ] Implement `Trigger` enum variants
- [ ] Build trigger evaluation engine
- [ ] Support boolean combinations (AND, OR, NOT)
- [ ] Add trigger debugging/logging

#### 3.3 Prompt Composition Engine
- [ ] Implement priority-based ordering
- [ ] Handle mutual exclusions
- [ ] Resolve dependency chains
- [ ] Build final prompt assembly:
  ```rust
  struct ComposedPrompt {
      system: String,
      context: String,
      query: String,
      examples: Vec<String>,
      total_tokens: usize,
  }
  ```
- [ ] Add token budget management (fit within model limits)

#### 3.4 Few-Shot Example Selection
- [ ] Store example library with embeddings
- [ ] Select examples based on query similarity
- [ ] Filter by intent match
- [ ] Limit by token budget

---

### Phase 4: Modular Context Engine

#### 4.1 Context Object Store
- [ ] Implement `ContextObject` struct
- [ ] Build in-memory store with persistence (SQLite or sled)
- [ ] Add indexing by:
  - [ ] ID
  - [ ] Embedding (vector index)
  - [ ] Type
  - [ ] Topics
  - [ ] Timestamps
- [ ] Implement CRUD operations

#### 4.2 Context Partitioner (Key Innovation)
- [ ] Design partitioning prompt for small LLM
- [ ] Implement streaming partitioner for long contexts
- [ ] Define partition boundaries:
  - [ ] Topic shifts
  - [ ] Speaker changes
  - [ ] Logical segments (instructions, facts, examples)
- [ ] Preserve cross-references between partitions
- [ ] Validate losslessness (round-trip test)

```rust
struct ContextPartitioner {
    llm: Arc<dyn LlmProvider>,
    partitioning_prompt: PromptTemplate,
    max_object_tokens: usize,
    min_object_tokens: usize,
}

impl ContextPartitioner {
    async fn partition(&self, raw_context: &str) -> Result<Vec<ContextObject>>;
    async fn validate_lossless(&self, original: &str, objects: &[ContextObject]) -> bool;
}
```

#### 4.3 Relevance Scoring
- [ ] Implement multi-factor scoring:
  ```rust
  fn score_relevance(co: &ContextObject, query: &QueryAnalysis) -> f32 {
      let semantic_score = cosine_similarity(&co.embedding, &query.embedding);
      let recency_score = recency_decay(co.last_accessed);
      let reference_score = reference_importance(co);
      let type_match_score = type_relevance(co.object_type, &query.required_context_types);

      // Weighted combination
      0.4 * semantic_score + 0.2 * recency_score + 0.2 * reference_score + 0.2 * type_match_score
  }
  ```
- [ ] Tune weights empirically
- [ ] Add query-specific weight overrides

#### 4.4 Context Composer
- [ ] Select top-K context objects by score
- [ ] Respect token budget
- [ ] Order by logical coherence (not just score)
- [ ] Handle cross-references (include dependencies)
- [ ] Format for injection into prompt

#### 4.5 Context Compression (Optional Enhancement)
- [ ] Implement summarization for low-priority objects
- [ ] Store both full and compressed versions
- [ ] Use compression for objects beyond recency threshold
- [ ] Preserve original for potential re-expansion

---

### Phase 5: Agent Orchestration

#### 5.1 Main Pipeline
- [ ] Implement request lifecycle:
  ```rust
  async fn process_request(&self, user_input: &str) -> Result<AgentResponse> {
      // 1. Analyze query
      let analysis = self.analyzer.analyze(user_input).await?;

      // 2. Build dynamic prompt
      let modules = self.prompt_builder.select_modules(&analysis)?;

      // 3. Retrieve relevant context
      let context_objects = self.context_engine.retrieve(&analysis).await?;

      // 4. Compose final prompt
      let prompt = self.prompt_builder.compose(modules, context_objects, &analysis)?;

      // 5. Call primary LLM
      let response = self.large_llm.complete(&prompt).await?;

      // 6. Update context store
      self.context_engine.ingest(&user_input, &response).await?;

      Ok(AgentResponse { content: response, metadata: ... })
  }
  ```
- [ ] Add observability (trace each step)
- [ ] Implement error recovery
- [ ] Add request timeout handling

#### 5.2 State Management
- [ ] Design conversation state struct
- [ ] Implement session persistence
- [ ] Handle multi-turn context accumulation
- [ ] Add session cleanup/archival

#### 5.3 Configuration System
- [ ] TOML/YAML config for:
  - [ ] Model endpoints and API keys
  - [ ] Token budgets
  - [ ] Scoring weights
  - [ ] Module paths
- [ ] Environment variable overrides
- [ ] Runtime configuration updates

---

### Phase 6: Testing & Evaluation

#### 6.1 Unit Tests
- [ ] Query analyzer accuracy tests
- [ ] Prompt module trigger tests
- [ ] Context partitioner losslessness tests
- [ ] Relevance scorer tests

#### 6.2 Integration Tests
- [ ] Full pipeline end-to-end tests
- [ ] Multi-turn conversation tests
- [ ] Context overflow handling tests
- [ ] Error recovery tests

#### 6.3 Benchmark Suite
- [ ] Latency benchmarks (analyzer, composer, full pipeline)
- [ ] Token efficiency benchmarks (vs static prompts)
- [ ] Memory usage profiling
- [ ] Comparison with baseline (no dynamic prompt/context)

#### 6.4 Evaluation Metrics
- [ ] **Context Recall**: Do we retrieve the right COs?
- [ ] **Prompt Relevance**: Do activated modules improve output?
- [ ] **Losslessness**: Can we reconstruct original context from COs?
- [ ] **Latency Overhead**: Cost of small LLM analysis
- [ ] **Response Quality**: Human eval or LLM-as-judge

---

### Phase 7: Demo & Documentation

#### 7.1 Interactive Demo
- [ ] CLI interface for real-time interaction
- [ ] Visualization of:
  - [ ] Query analysis results
  - [ ] Activated prompt modules
  - [ ] Retrieved context objects
  - [ ] Final composed prompt
- [ ] Side-by-side comparison mode (dynamic vs static)

#### 7.2 Thesis Documentation
- [ ] Literature review chapter
- [ ] Architecture design chapter
- [ ] Implementation details chapter
- [ ] Evaluation results chapter
- [ ] Future work discussion

#### 7.3 Code Documentation
- [ ] Rustdoc for all public APIs
- [ ] Architecture decision records (ADRs)
- [ ] Example usage guides
- [ ] Configuration reference

---

## Key Design Decisions

### Decision 1: Small LLM Selection

**Options:**
| Model | Tokens/sec | Quality | Local? | Cost |
|-------|-----------|---------|--------|------|
| Qwen2.5-3B | ~100 | Good | Yes | Free |
| Phi-3-mini | ~80 | Good | Yes | Free |
| Claude Haiku | ~150 | Excellent | No | $0.25/1M |
| GPT-4o-mini | ~100 | Excellent | No | $0.15/1M |

**Recommendation:** Start with Claude Haiku for quality, add local Qwen2.5-3B for cost-sensitive deployments.

### Decision 2: Context Partitioning Strategy

**Options:**
1. **LLM-based**: Small LLM identifies boundaries (most flexible, higher latency)
2. **Heuristic**: Rule-based splitting (fast, less accurate)
3. **Hybrid**: Heuristics for initial split, LLM for refinement

**Recommendation:** Hybrid approach—use heuristics for obvious boundaries, LLM for ambiguous cases.

### Decision 3: Vector Store Backend

**Options:**
| Backend | Rust Support | Performance | Features |
|---------|--------------|-------------|----------|
| In-memory HNSW | Excellent | Fast | Basic |
| usearch | Excellent | Very fast | Good |
| Qdrant | Good | Fast | Full-featured |
| SQLite + vec | Good | Moderate | Simple |

**Recommendation:** Start with in-memory HNSW, migrate to Qdrant for production scale.

### Decision 4: Persistence Strategy

**Options:**
1. **SQLite**: Simple, single-file, good for prototyping
2. **sled**: Embedded, Rust-native, fast
3. **PostgreSQL**: Full-featured, production-ready

**Recommendation:** sled for embedded use, PostgreSQL for distributed deployment.

---

## Success Criteria

### Minimum Viable Thesis

1. ✅ Query analyzer correctly classifies intent >85% of the time
2. ✅ Dynamic prompt modules activate appropriately in >90% of cases
3. ✅ Context partitioning is lossless (100% reconstruction)
4. ✅ Retrieval precision@5 > 0.7 for relevant context
5. ✅ End-to-end latency < 500ms (excluding primary LLM)
6. ✅ Demonstrable improvement in response quality vs static baseline

### Stretch Goals

- [ ] Sub-100ms analysis latency with local small LLM
- [ ] Support for 1M+ token context archives
- [ ] Multi-modal context objects (images, code, structured data)
- [ ] Federated context across multiple sessions/users
- [ ] Self-improving module selection via feedback

---

## Timeline-Free Milestones

The following milestones represent logical checkpoints, not time estimates:

1. **M1: Foundation Complete** — LLM interfaces working, basic embedding pipeline
2. **M2: Analyzer Working** — Query analysis producing valid structured output
3. **M3: Prompt Builder Working** — Modules activating based on analysis
4. **M4: Context Engine Working** — Partitioning, storing, retrieving COs
5. **M5: Integration Complete** — Full pipeline operational
6. **M6: Evaluated** — Benchmarks run, metrics collected
7. **M7: Thesis Ready** — Documentation complete, demo polished

---

## References

### Foundational Research
- [LLM Agents - Prompt Engineering Guide](https://www.promptingguide.ai/research/llm-agents)
- [Awesome Agent Papers](https://github.com/luo-junyu/Awesome-Agent-Papers)
- [Autonomous Agents Research](https://github.com/tmgthb/Autonomous-Agents)

### Context Management
- [Context Engineering for Agents - LangChain](https://www.blog.langchain.com/context-engineering-for-agents/)
- [Mem0: Scalable Long-Term Memory](https://arxiv.org/pdf/2504.19413)
- [Top Techniques to Manage Context Lengths](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
- [Context Windows Explained](https://unstructured.io/insights/llm-context-windows-explained-a-developer-s-guide)

### Compression & Memory
- [KVzip Context Compression](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html)
- [DFloat11 Lossless Compression](https://arxiv.org/html/2504.11651v1)
- [TTT-E2E Test-Time Training](https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time/)
- [Awesome LLM Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)

### Prompt Engineering
- [Dynamic Prompt Composition at Scale](https://overctrl.com/prompt-engineering-at-scale-building-robust-prompt-pipelines-caching-dynamic-composition-and-evaluation-for-llms-in-production/)
- [Prompt Templates Analysis](https://arxiv.org/html/2504.02052v2)
- [Automatic Prompt Engineering](https://arxiv.org/html/2401.14423v4)

### Routing & Orchestration
- [Confidence-Aware Routing](https://arxiv.org/html/2601.04861)
- [LLM Orchestration Frameworks](https://www.zenml.io/blog/best-llm-orchestration-frameworks)
- [Multi-Agent Orchestration Analysis](https://medium.com/@frankmorales_91352/multi-agent-orchestration-and-the-future-of-llm-specialization-an-analysis-of-the-ce86abd731cc)

### Rust Frameworks
- [Rig - LLM Applications in Rust](https://rig.rs/)
- [graph-flow (rs-graph-llm)](https://github.com/a-agmon/rs-graph-llm)
- [Kowalski Agentic Framework](https://dev.to/yarenty/kowalski-the-rust-native-agentic-ai-framework-53k4)
- [AutoAgents Multi-Agent Framework](https://github.com/liquidos-ai/AutoAgents)
- [Awesome Rust LLM](https://github.com/jondot/awesome-rust-llm)

---

## Appendix A: Example Prompt Module Definitions

```toml
# modules/coding_mode.toml
[module]
id = "coding_mode"
priority = 100

[module.content]
text = """
You are an expert software engineer. When writing code:
- Prefer clarity over cleverness
- Include error handling
- Follow language idioms
- Explain non-obvious decisions
"""

[[module.triggers]]
type = "keyword_present"
keywords = ["code", "function", "implement", "bug", "error", "compile"]

[[module.triggers]]
type = "intent_match"
intent = "Command"

[module.metadata]
mutual_exclusions = ["casual_chat_mode"]
```

```toml
# modules/memory_recall.toml
[module]
id = "memory_recall"
priority = 50

[module.content]
text = """
The user is referencing previous context. Relevant information from earlier:
{{#each context_objects}}
- {{this.summary}}
{{/each}}
"""

[[module.triggers]]
type = "keyword_present"
keywords = ["remember", "earlier", "before", "last time", "you said"]

[[module.triggers]]
type = "memory_trigger_detected"
min_confidence = 0.7
```

---

## Appendix B: Context Partitioning Prompt

```
You are a context partitioning assistant. Your task is to analyze the following conversation/document and split it into discrete, self-contained context objects.

Rules:
1. Each object should be semantically coherent (one topic/concept)
2. Preserve all information—this must be lossless
3. Mark relationships between objects (references, dependencies)
4. Classify each object by type: Fact, Instruction, Example, Conversation, ToolResult, SystemState
5. Extract key topics for each object

Input:
{raw_context}

Output format (JSON):
{
  "objects": [
    {
      "content": "...",
      "type": "Fact|Instruction|Example|Conversation|ToolResult|SystemState",
      "topics": ["topic1", "topic2"],
      "references": ["id_of_related_object"],
      "importance": 0.0-1.0
    }
  ]
}
```

---

*This plan is a living document. Update as the implementation progresses.*
