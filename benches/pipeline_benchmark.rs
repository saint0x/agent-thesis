//! Benchmarks for the Dynamic Prompt Agent pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use dynamic_prompt_agent::{
    analyzer::QueryAnalysis,
    prompt::{PromptBuilder, ModuleConfig, create_default_modules},
    context::{ContextObject, ContextType, RelevanceScorer, ScoringWeights},
    embeddings::{cosine_similarity, VectorStore, VectorEntry},
};

fn benchmark_cosine_similarity(c: &mut Criterion) {
    let dim = 384;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

    c.bench_function("cosine_similarity_384d", |bencher| {
        bencher.iter(|| {
            cosine_similarity(black_box(&a), black_box(&b))
        });
    });
}

fn benchmark_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");

    for size in [100, 1000, 10000].iter() {
        let store = VectorStore::new(384);

        // Populate store
        for i in 0..*size {
            let embedding: Vec<f32> = (0..384).map(|j| ((i + j) as f32).sin()).collect();
            let entry = VectorEntry {
                id: uuid::Uuid::new_v4(),
                embedding,
                metadata: std::collections::HashMap::new(),
            };
            store.insert(entry).unwrap();
        }

        // Rebuild index
        store.rebuild_index().unwrap();

        let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.5).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("search_top_5", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    store.search(black_box(&query), 5).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_relevance_scoring(c: &mut Criterion) {
    let scorer = RelevanceScorer::new()
        .with_weights(ScoringWeights::default())
        .with_recency_half_life(24.0);

    let objects: Vec<ContextObject> = (0..100)
        .map(|i| {
            let mut obj = ContextObject::new(
                format!("Context object {}", i),
                ContextType::Fact,
            );
            obj.topics = vec![format!("topic{}", i % 10)];
            obj.importance = (i as f32 / 100.0).sin().abs();
            obj.embedding = Some((0..384).map(|j| ((i + j) as f32).sin()).collect());
            obj
        })
        .collect();

    let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.3).sin()).collect();
    let query_topics = vec!["topic5".to_string()];

    c.bench_function("relevance_scoring_100_objects", |bencher| {
        bencher.iter(|| {
            scorer.rank(
                black_box(&objects),
                Some(black_box(&query_embedding)),
                black_box(&query_topics),
            )
        });
    });
}

fn benchmark_prompt_composition(c: &mut Criterion) {
    let config = ModuleConfig {
        base_system_prompt: "You are a helpful assistant.".to_string(),
        ..Default::default()
    };

    let mut builder = PromptBuilder::new(config);
    builder.register_modules(create_default_modules());

    let analysis = QueryAnalysis {
        original_query: "How do I implement a sorting algorithm in Rust?".to_string(),
        primary_intent: dynamic_prompt_agent::analyzer::Intent::Command,
        topics: vec!["programming".to_string(), "rust".to_string()],
        complexity: 0.6,
        requires_reasoning: true,
        ..Default::default()
    };

    let context_objects: Vec<ContextObject> = (0..5)
        .map(|i| {
            ContextObject::new(
                format!("Relevant context {}", i),
                ContextType::Fact,
            )
        })
        .collect();

    c.bench_function("prompt_composition", |bencher| {
        bencher.iter(|| {
            builder.compose(
                black_box(&analysis),
                black_box(&context_objects),
                None,
            ).unwrap()
        });
    });
}

fn benchmark_trigger_evaluation(c: &mut Criterion) {
    use dynamic_prompt_agent::prompt::{Trigger, TriggerEvaluator};

    let evaluator = TriggerEvaluator::new().with_conversation_length(5);

    let analysis = QueryAnalysis {
        original_query: "Please implement a function to sort an array".to_string(),
        primary_intent: dynamic_prompt_agent::analyzer::Intent::Command,
        topics: vec!["programming".to_string()],
        sentiment: dynamic_prompt_agent::analyzer::Sentiment {
            score: 0.5,
            magnitude: 0.3,
            label: dynamic_prompt_agent::analyzer::types::SentimentLabel::Positive,
        },
        complexity: 0.7,
        requires_reasoning: true,
        ..Default::default()
    };

    let trigger = Trigger::And {
        triggers: vec![
            Trigger::KeywordPresent {
                keywords: vec!["implement".to_string(), "function".to_string()],
                case_sensitive: false,
            },
            Trigger::IntentMatch {
                intents: vec!["command".to_string()],
            },
            Trigger::ComplexityRange { min: 0.5, max: 1.0 },
        ],
    };

    c.bench_function("trigger_evaluation_complex", |bencher| {
        bencher.iter(|| {
            evaluator.evaluate(black_box(&trigger), black_box(&analysis))
        });
    });
}

criterion_group!(
    benches,
    benchmark_cosine_similarity,
    benchmark_vector_search,
    benchmark_relevance_scoring,
    benchmark_prompt_composition,
    benchmark_trigger_evaluation,
);

criterion_main!(benches);
