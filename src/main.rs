//! Dynamic Prompt Agent CLI
//!
//! A command-line interface for the Dynamic Prompt Agent.

use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use dynamic_prompt_agent::{
    Agent, AgentConfig, AgentResponse, Result,
    prompt::create_default_modules,
};

#[derive(Parser)]
#[command(name = "dpa")]
#[command(about = "Dynamic Prompt Agent - Runtime-adaptive prompt composition and modular context management")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable debug output
    #[arg(short, long, global = true)]
    debug: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// API key (or use ANTHROPIC_API_KEY/OPENAI_API_KEY env var)
    #[arg(short = 'k', long, global = true)]
    api_key: Option<String>,

    /// Small model for analysis (e.g., claude-3-haiku-20240307)
    #[arg(long, global = true)]
    small_model: Option<String>,

    /// Large model for generation (e.g., claude-3-5-sonnet-20241022)
    #[arg(long, global = true)]
    large_model: Option<String>,

    /// Provider (anthropic or openai)
    #[arg(short, long, global = true, default_value = "anthropic")]
    provider: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat {
        /// Session ID (generates UUID if not provided)
        #[arg(short, long)]
        session: Option<String>,

        /// System prompt
        #[arg(short = 'p', long)]
        system_prompt: Option<String>,

        /// Persistence path for context storage
        #[arg(long)]
        persist: Option<String>,
    },

    /// Process a single message
    Query {
        /// The message to process
        message: String,

        /// System prompt
        #[arg(short = 'p', long)]
        system_prompt: Option<String>,
    },

    /// Analyze a query without generating a response
    Analyze {
        /// The query to analyze
        query: String,
    },

    /// Show information about the agent
    Info,

    /// List available prompt modules
    Modules,

    /// Run a benchmark
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();

    // Load environment variables
    dotenvy::dotenv().ok();

    match &cli.command {
        Some(Commands::Chat { session, system_prompt, persist }) => {
            run_chat(&cli, session.clone(), system_prompt.clone(), persist.clone()).await
        }
        Some(Commands::Query { message, system_prompt }) => {
            run_query(&cli, message, system_prompt.clone()).await
        }
        Some(Commands::Analyze { query }) => {
            run_analyze(&cli, query).await
        }
        Some(Commands::Info) => {
            show_info(&cli).await
        }
        Some(Commands::Modules) => {
            list_modules()
        }
        Some(Commands::Bench { iterations }) => {
            run_benchmark(&cli, *iterations).await
        }
        None => {
            // Default to chat mode
            run_chat(&cli, None, None, None).await
        }
    }
}

async fn create_agent(cli: &Cli, system_prompt: Option<String>, persist: Option<String>, session: Option<String>) -> Result<Agent> {
    let mut config = AgentConfig::default();

    // Set provider
    config.small_llm.provider = cli.provider.clone();
    config.large_llm.provider = cli.provider.clone();

    // Set models if specified
    if let Some(ref model) = cli.small_model {
        config.small_llm.model = model.clone();
    } else if cli.provider == "openai" {
        config.small_llm.model = "gpt-4o-mini".to_string();
    }

    if let Some(ref model) = cli.large_model {
        config.large_llm.model = model.clone();
    } else if cli.provider == "openai" {
        config.large_llm.model = "gpt-4o".to_string();
    }

    // Set API key if provided
    if let Some(ref key) = cli.api_key {
        config.small_llm.api_key = Some(key.clone());
        config.large_llm.api_key = Some(key.clone());
    }

    // Set system prompt
    if let Some(prompt) = system_prompt {
        config.prompt_config.base_system_prompt = prompt;
    } else {
        config.prompt_config.base_system_prompt = "You are a helpful AI assistant with excellent memory and reasoning capabilities.".to_string();
    }

    // Set persistence path
    config.persistence_path = persist;

    // Set session ID
    config.session_id = session;

    // Set debug mode
    config.debug = cli.debug;

    Agent::new(config).await
}

async fn run_chat(
    cli: &Cli,
    session: Option<String>,
    system_prompt: Option<String>,
    persist: Option<String>,
) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Dynamic Prompt Agent - Interactive Chat            ║");
    println!("║                                                              ║");
    println!("║  Commands:                                                   ║");
    println!("║    /quit or /exit  - Exit the chat                          ║");
    println!("║    /reset          - Reset conversation                      ║");
    println!("║    /status         - Show session status                     ║");
    println!("║    /context        - Show context summary                    ║");
    println!("║    /debug          - Toggle debug output                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut agent = create_agent(cli, system_prompt, persist, session).await?;
    let (small_info, large_info) = agent.get_model_info();

    println!("Session: {}", agent.state().session_id);
    println!("Small model: {} ({})", small_info.model, small_info.provider);
    println!("Large model: {} ({})", large_info.model, large_info.provider);
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut debug_mode = cli.debug;

    loop {
        print!("You: ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" => {
                    println!("Goodbye!");
                    break;
                }
                "/reset" => {
                    agent.reset()?;
                    println!("Conversation reset.");
                    continue;
                }
                "/status" => {
                    let state = agent.state();
                    println!("\nSession Status:");
                    println!("  Session ID: {}", state.session_id);
                    println!("  Turn number: {}", state.turn_number);
                    println!("  Recent exchanges: {}", state.recent_exchanges.len());
                    println!("  Duration: {:?}", state.duration());
                    println!();
                    continue;
                }
                "/context" => {
                    match agent.context_summary() {
                        Ok(summary) => {
                            println!("\nContext Summary:");
                            println!("  Total objects: {}", summary.total_objects);
                            println!("  Total tokens: {}", summary.total_tokens);
                            println!("  Top topics: {:?}", summary.top_topics);
                            println!("  Types: {:?}", summary.type_counts);
                            println!();
                        }
                        Err(e) => println!("Error getting context: {}", e),
                    }
                    continue;
                }
                "/debug" => {
                    debug_mode = !debug_mode;
                    println!("Debug mode: {}", if debug_mode { "ON" } else { "OFF" });
                    continue;
                }
                _ => {
                    println!("Unknown command: {}", input);
                    continue;
                }
            }
        }

        // Process the message
        match agent.process(input).await {
            Ok(response) => {
                println!("\nAssistant: {}\n", response.content);

                if debug_mode {
                    print_debug_info(&response);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    // Flush any pending writes
    agent.flush()?;

    Ok(())
}

async fn run_query(cli: &Cli, message: &str, system_prompt: Option<String>) -> Result<()> {
    let mut agent = create_agent(cli, system_prompt, None, None).await?;

    let response = agent.process(message).await?;

    println!("{}", response.content);

    if cli.debug {
        eprintln!();
        print_debug_info(&response);
    }

    Ok(())
}

async fn run_analyze(cli: &Cli, query: &str) -> Result<()> {
    println!("Analyzing query: \"{}\"", query);
    println!();

    // Create LLM config for analysis
    let mut config = AgentConfig::default();
    config.small_llm.provider = cli.provider.clone();
    if let Some(ref key) = cli.api_key {
        config.small_llm.api_key = Some(key.clone());
    }
    if let Some(ref model) = cli.small_model {
        config.small_llm.model = model.clone();
    } else if cli.provider == "openai" {
        config.small_llm.model = "gpt-4o-mini".to_string();
    }

    let small_llm = dynamic_prompt_agent::llm::create_provider(&config.small_llm).await?;

    let analyzer = dynamic_prompt_agent::analyzer::QueryAnalyzer::new(small_llm.into());
    let analysis = analyzer.analyze(query, None).await?;

    println!("Analysis Results:");
    println!("================");
    println!();
    println!("Intent: {:?}", analysis.primary_intent);
    println!("Confidence: {:.2}", analysis.confidence);
    println!();
    println!("Sentiment:");
    println!("  Score: {:.2}", analysis.sentiment.score);
    println!("  Magnitude: {:.2}", analysis.sentiment.magnitude);
    println!("  Label: {:?}", analysis.sentiment.label);
    println!();
    println!("Keywords: {:?}", analysis.keywords.iter().map(|k| &k.text).collect::<Vec<_>>());
    println!("Topics: {:?}", analysis.topics);
    println!();
    println!("Complexity: {:.2}", analysis.complexity);
    println!("Requires knowledge: {}", analysis.requires_knowledge);
    println!("Requires reasoning: {}", analysis.requires_reasoning);
    println!();

    if !analysis.memory_triggers.is_empty() {
        println!("Memory triggers:");
        for trigger in &analysis.memory_triggers {
            println!("  - \"{}\" ({:?}, confidence: {:.2})",
                trigger.trigger_phrase, trigger.trigger_type, trigger.confidence);
        }
        println!();
    }

    if !analysis.suggested_modules.is_empty() {
        println!("Suggested modules:");
        for module in &analysis.suggested_modules {
            println!("  - {} (confidence: {:.2}): {}", module.module_id, module.confidence, module.reason);
        }
    }

    if let Some(latency) = analysis.latency_ms {
        println!();
        println!("Analysis latency: {}ms", latency);
    }

    Ok(())
}

async fn show_info(cli: &Cli) -> Result<()> {
    println!("Dynamic Prompt Agent");
    println!("====================");
    println!();
    println!("A runtime-adaptive prompt composition and modular context management agent.");
    println!();
    println!("Architecture:");
    println!("  1. Query Analyzer - Small LLM analyzes queries for intent, sentiment, keywords");
    println!("  2. Dynamic Prompt Builder - Composes prompts based on analysis");
    println!("  3. Modular Context Engine - Partitions and retrieves context objects");
    println!("  4. Agent Pipeline - Orchestrates the full request lifecycle");
    println!();

    let config = AgentConfig::default();
    println!("Default Configuration:");
    println!("  Small LLM: {} ({})", config.small_llm.model, config.small_llm.provider);
    println!("  Large LLM: {} ({})", config.large_llm.model, config.large_llm.provider);
    println!("  Context token budget: {}", config.context_config.token_budget);
    println!("  Top-K retrieval: {}", config.context_config.top_k);
    println!("  Embedding dimension: {}", config.embedding_dim);
    println!();

    let modules = create_default_modules();
    println!("Built-in Prompt Modules: {}", modules.len());
    for module in &modules {
        println!("  - {} (priority: {})", module.id, module.priority);
    }

    Ok(())
}

fn list_modules() -> Result<()> {
    let modules = create_default_modules();

    println!("Available Prompt Modules");
    println!("========================");
    println!();

    for module in modules {
        println!("Module: {}", module.id);
        println!("  Name: {}", module.name);
        println!("  Description: {}", module.description);
        println!("  Priority: {}", module.priority);
        println!("  Section: {:?}", module.section);
        println!("  Triggers: {} defined", module.triggers.len());
        println!();
    }

    Ok(())
}

async fn run_benchmark(cli: &Cli, iterations: usize) -> Result<()> {
    println!("Running benchmark with {} iterations...", iterations);
    println!();

    let agent = create_agent(cli, None, None, None).await?;

    let test_queries = vec![
        "What is the capital of France?",
        "Implement a quicksort algorithm in Rust",
        "Explain the difference between TCP and UDP",
        "Remember what we discussed earlier about memory management",
        "Can you help me analyze this data?",
    ];

    let mut total_analysis_time = 0u64;
    let mut total_composition_time = 0u64;

    for i in 0..iterations {
        let query = &test_queries[i % test_queries.len()];

        // Benchmark analysis
        let start = std::time::Instant::now();
        let config = AgentConfig::default();
        let small_llm = dynamic_prompt_agent::llm::create_provider(&config.small_llm).await?;
        let analyzer = dynamic_prompt_agent::analyzer::QueryAnalyzer::new(small_llm.into());
        let analysis = analyzer.analyze(query, None).await?;
        let analysis_time = start.elapsed().as_millis() as u64;
        total_analysis_time += analysis_time;

        // Benchmark composition
        let start = std::time::Instant::now();
        let prompt_builder = dynamic_prompt_agent::prompt::PromptBuilder::new(Default::default());
        let composed = prompt_builder.compose(&analysis, &[], None)?;
        let composition_time = start.elapsed().as_millis() as u64;
        total_composition_time += composition_time;

        if cli.debug {
            println!("  Iteration {}: analysis={}ms, composition={}ms",
                i + 1, analysis_time, composition_time);
        }
    }

    println!("Benchmark Results:");
    println!("==================");
    println!("  Iterations: {}", iterations);
    println!("  Avg analysis time: {:.2}ms", total_analysis_time as f64 / iterations as f64);
    println!("  Avg composition time: {:.2}ms", total_composition_time as f64 / iterations as f64);
    println!("  Total time: {}ms", total_analysis_time + total_composition_time);

    Ok(())
}

fn print_debug_info(response: &AgentResponse) {
    eprintln!("─── Debug Info ───────────────────────────────────────");
    eprintln!("Turn: {}", response.state_snapshot.turn_number);
    eprintln!("Analysis:");
    eprintln!("  Intent: {:?}", response.analysis.primary_intent);
    eprintln!("  Confidence: {:.2}", response.analysis.confidence);
    eprintln!("  Topics: {:?}", response.analysis.topics);
    eprintln!();
    eprintln!("Prompt Composition:");
    eprintln!("  Activated modules: {:?}", response.composed_prompt.activated_modules);
    eprintln!("  Estimated tokens: {}", response.composed_prompt.estimated_tokens);
    eprintln!();
    eprintln!("Metrics:");
    eprintln!("  Total time: {}ms", response.metrics.total_time_ms);
    eprintln!("  Analysis: {}ms", response.metrics.analysis_time_ms);
    eprintln!("  Retrieval: {}ms", response.metrics.retrieval_time_ms);
    eprintln!("  Composition: {}ms", response.metrics.composition_time_ms);
    eprintln!("  Generation: {}ms", response.metrics.generation_time_ms);
    eprintln!("  Context objects: {}", response.metrics.context_objects_retrieved);
    eprintln!("  Total tokens: {}", response.metrics.total_tokens());
    eprintln!("──────────────────────────────────────────────────────");
    eprintln!();
}
