//! Vector store with HNSW indexing for fast similarity search

use instant_distance::{Builder, HnswMap, Search};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use uuid::Uuid;

use super::similarity::cosine_similarity;
use crate::{DpaError, Result};

/// Entry in the vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Point wrapper for HNSW
#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Use 1 - cosine_similarity as distance (so higher similarity = lower distance)
        1.0 - cosine_similarity(&self.0, &other.0)
    }
}

/// Vector store with HNSW indexing
pub struct VectorStore {
    dimension: usize,
    entries: RwLock<HashMap<Uuid, VectorEntry>>,
    index: RwLock<Option<HnswMap<Point, Uuid>>>,
    needs_rebuild: RwLock<bool>,
}

impl VectorStore {
    /// Create a new vector store with the given embedding dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: RwLock::new(HashMap::new()),
            index: RwLock::new(None),
            needs_rebuild: RwLock::new(false),
        }
    }

    /// Insert an entry into the store
    pub fn insert(&self, entry: VectorEntry) -> Result<()> {
        if entry.embedding.len() != self.dimension {
            return Err(DpaError::EmbeddingError(format!(
                "Expected dimension {}, got {}",
                self.dimension,
                entry.embedding.len()
            )));
        }

        let mut entries = self.entries.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;

        entries.insert(entry.id, entry);

        let mut needs_rebuild = self.needs_rebuild.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;
        *needs_rebuild = true;

        Ok(())
    }

    /// Insert multiple entries
    pub fn insert_batch(&self, batch: Vec<VectorEntry>) -> Result<()> {
        let mut entries = self.entries.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;

        for entry in batch {
            if entry.embedding.len() != self.dimension {
                return Err(DpaError::EmbeddingError(format!(
                    "Expected dimension {}, got {}",
                    self.dimension,
                    entry.embedding.len()
                )));
            }
            entries.insert(entry.id, entry);
        }

        let mut needs_rebuild = self.needs_rebuild.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;
        *needs_rebuild = true;

        Ok(())
    }

    /// Remove an entry from the store
    pub fn remove(&self, id: &Uuid) -> Result<Option<VectorEntry>> {
        let mut entries = self.entries.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;

        let removed = entries.remove(id);

        if removed.is_some() {
            let mut needs_rebuild = self.needs_rebuild.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            *needs_rebuild = true;
        }

        Ok(removed)
    }

    /// Get an entry by ID
    pub fn get(&self, id: &Uuid) -> Result<Option<VectorEntry>> {
        let entries = self.entries.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(entries.get(id).cloned())
    }

    /// Rebuild the HNSW index
    pub fn rebuild_index(&self) -> Result<()> {
        let entries = self.entries.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        if entries.is_empty() {
            let mut index = self.index.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            *index = None;
            return Ok(());
        }

        let (points, values): (Vec<Point>, Vec<Uuid>) = entries
            .values()
            .map(|e| (Point(e.embedding.clone()), e.id))
            .unzip();

        let hnsw = Builder::default().build(points, values);

        let mut index = self.index.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;
        *index = Some(hnsw);

        let mut needs_rebuild = self.needs_rebuild.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;
        *needs_rebuild = false;

        Ok(())
    }

    /// Search for the k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        if query.len() != self.dimension {
            return Err(DpaError::EmbeddingError(format!(
                "Expected dimension {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        // Rebuild index if needed
        let needs_rebuild = *self.needs_rebuild.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        if needs_rebuild {
            self.rebuild_index()?;
        }

        let index = self.index.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        let entries = self.entries.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        match &*index {
            Some(hnsw) => {
                let mut search = Search::default();
                let query_point = Point(query.to_vec());

                let results: Vec<(Uuid, f32)> = hnsw
                    .search(&query_point, &mut search)
                    .take(k)
                    .map(|item| {
                        let similarity = 1.0 - item.distance;
                        (*item.value, similarity)
                    })
                    .collect();

                Ok(results)
            }
            None => {
                // Fallback to brute-force search if index is empty
                let mut results: Vec<(Uuid, OrderedFloat<f32>)> = entries
                    .values()
                    .map(|e| {
                        let sim = cosine_similarity(&e.embedding, query);
                        (e.id, OrderedFloat(sim))
                    })
                    .collect();

                results.sort_by(|a, b| b.1.cmp(&a.1));
                results.truncate(k);

                Ok(results.into_iter().map(|(id, sim)| (id, sim.0)).collect())
            }
        }
    }

    /// Get the number of entries in the store
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all entry IDs
    pub fn all_ids(&self) -> Result<Vec<Uuid>> {
        let entries = self.entries.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(entries.keys().copied().collect())
    }
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("dimension", &self.dimension)
            .field("entries_count", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_insert_search() {
        let store = VectorStore::new(3);

        let entry1 = VectorEntry {
            id: Uuid::new_v4(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        };

        let entry2 = VectorEntry {
            id: Uuid::new_v4(),
            embedding: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        };

        store.insert(entry1.clone()).unwrap();
        store.insert(entry2.clone()).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, entry1.id);
    }
}
