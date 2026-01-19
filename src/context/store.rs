//! Context object storage with persistence and indexing

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use tracing::{debug, info, warn};

use super::types::{ContextObject, ContextType};
use crate::embeddings::{VectorStore, VectorEntry, EmbeddingProvider};
use crate::{DpaError, Result};

/// Persistent context store with vector indexing
pub struct ContextStore {
    /// In-memory object storage
    objects: RwLock<HashMap<Uuid, ContextObject>>,

    /// Vector store for similarity search
    vector_store: VectorStore,

    /// Embedding provider
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Sled database for persistence (optional)
    db: Option<sled::Db>,

    /// Index by type
    type_index: RwLock<HashMap<ContextType, Vec<Uuid>>>,

    /// Index by topic
    topic_index: RwLock<HashMap<String, Vec<Uuid>>>,
}

impl ContextStore {
    /// Create a new in-memory context store
    pub fn new(embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        let dimension = embedding_provider.dimension();
        Self {
            objects: RwLock::new(HashMap::new()),
            vector_store: VectorStore::new(dimension),
            embedding_provider,
            db: None,
            type_index: RwLock::new(HashMap::new()),
            topic_index: RwLock::new(HashMap::new()),
        }
    }

    /// Create a persistent context store
    pub fn with_persistence<P: AsRef<Path>>(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        path: P,
    ) -> Result<Self> {
        let dimension = embedding_provider.dimension();
        let db = sled::open(path.as_ref())
            .map_err(|e| DpaError::ContextError(format!("Failed to open database: {}", e)))?;

        let mut store = Self {
            objects: RwLock::new(HashMap::new()),
            vector_store: VectorStore::new(dimension),
            embedding_provider,
            db: Some(db),
            type_index: RwLock::new(HashMap::new()),
            topic_index: RwLock::new(HashMap::new()),
        };

        // Load existing data
        store.load_from_disk()?;

        Ok(store)
    }

    /// Insert a context object
    pub async fn insert(&self, mut object: ContextObject) -> Result<Uuid> {
        let id = object.id;

        // Generate embedding if not present
        if object.embedding.is_none() {
            let embeddings = self
                .embedding_provider
                .embed(&[object.content.clone()])
                .await?;
            if let Some(emb) = embeddings.into_iter().next() {
                object.embedding = Some(emb);
            }
        }

        // Insert into vector store
        if let Some(ref emb) = object.embedding {
            let entry = VectorEntry {
                id,
                embedding: emb.clone(),
                metadata: HashMap::new(),
            };
            self.vector_store.insert(entry)?;
        }

        // Update indexes
        {
            let mut type_idx = self.type_index.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            type_idx
                .entry(object.object_type)
                .or_default()
                .push(id);
        }

        {
            let mut topic_idx = self.topic_index.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            for topic in &object.topics {
                topic_idx
                    .entry(topic.to_lowercase())
                    .or_default()
                    .push(id);
            }
        }

        // Insert into main store
        {
            let mut objects = self.objects.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            objects.insert(id, object.clone());
        }

        // Persist to disk if enabled
        if let Some(ref db) = self.db {
            let serialized = serde_json::to_vec(&object)?;
            db.insert(id.as_bytes(), serialized)
                .map_err(|e| DpaError::ContextError(format!("Failed to persist: {}", e)))?;
        }

        debug!("Inserted context object: {}", id);
        Ok(id)
    }

    /// Insert multiple context objects
    pub async fn insert_batch(&self, objects: Vec<ContextObject>) -> Result<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(objects.len());
        for obj in objects {
            ids.push(self.insert(obj).await?);
        }
        Ok(ids)
    }

    /// Get a context object by ID
    pub fn get(&self, id: &Uuid) -> Result<Option<ContextObject>> {
        let objects = self.objects.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;
        Ok(objects.get(id).cloned())
    }

    /// Get a context object and mark it as accessed
    pub fn get_and_access(&self, id: &Uuid) -> Result<Option<ContextObject>> {
        let mut objects = self.objects.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;

        if let Some(obj) = objects.get_mut(id) {
            obj.mark_accessed();
            return Ok(Some(obj.clone()));
        }

        Ok(None)
    }

    /// Get multiple context objects by ID
    pub fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<ContextObject>> {
        let objects = self.objects.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(ids.iter().filter_map(|id| objects.get(id).cloned()).collect())
    }

    /// Update a context object
    pub fn update(&self, object: ContextObject) -> Result<()> {
        let id = object.id;

        let mut objects = self.objects.write().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
        })?;

        if !objects.contains_key(&id) {
            return Err(DpaError::ContextError(format!(
                "Context object not found: {}",
                id
            )));
        }

        objects.insert(id, object.clone());

        // Persist to disk if enabled
        if let Some(ref db) = self.db {
            let serialized = serde_json::to_vec(&object)?;
            db.insert(id.as_bytes(), serialized)
                .map_err(|e| DpaError::ContextError(format!("Failed to persist: {}", e)))?;
        }

        Ok(())
    }

    /// Remove a context object
    pub fn remove(&self, id: &Uuid) -> Result<Option<ContextObject>> {
        // Remove from vector store
        self.vector_store.remove(id)?;

        // Remove from main store
        let removed = {
            let mut objects = self.objects.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            objects.remove(id)
        };

        if let Some(ref obj) = removed {
            // Remove from type index
            {
                let mut type_idx = self.type_index.write().map_err(|e| {
                    DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
                })?;
                if let Some(ids) = type_idx.get_mut(&obj.object_type) {
                    ids.retain(|i| i != id);
                }
            }

            // Remove from topic index
            {
                let mut topic_idx = self.topic_index.write().map_err(|e| {
                    DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
                })?;
                for topic in &obj.topics {
                    if let Some(ids) = topic_idx.get_mut(&topic.to_lowercase()) {
                        ids.retain(|i| i != id);
                    }
                }
            }

            // Remove from disk if enabled
            if let Some(ref db) = self.db {
                db.remove(id.as_bytes())
                    .map_err(|e| DpaError::ContextError(format!("Failed to remove from disk: {}", e)))?;
            }
        }

        debug!("Removed context object: {}", id);
        Ok(removed)
    }

    /// Search for similar context objects
    pub fn search_similar(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        self.vector_store.search(query_embedding, k)
    }

    /// Search by embedding and return full objects
    pub fn search_similar_objects(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(ContextObject, f32)>> {
        let results = self.search_similar(query_embedding, k)?;

        let objects = self.objects.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(results
            .into_iter()
            .filter_map(|(id, score)| objects.get(&id).cloned().map(|obj| (obj, score)))
            .collect())
    }

    /// Get objects by type
    pub fn get_by_type(&self, object_type: ContextType) -> Result<Vec<ContextObject>> {
        let type_idx = self.type_index.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        let ids = type_idx.get(&object_type).cloned().unwrap_or_default();
        self.get_batch(&ids)
    }

    /// Get objects by topic
    pub fn get_by_topic(&self, topic: &str) -> Result<Vec<ContextObject>> {
        let topic_idx = self.topic_index.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        let ids = topic_idx.get(&topic.to_lowercase()).cloned().unwrap_or_default();
        self.get_batch(&ids)
    }

    /// Get all context objects
    pub fn get_all(&self) -> Result<Vec<ContextObject>> {
        let objects = self.objects.read().map_err(|e| {
            DpaError::ContextError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(objects.values().cloned().collect())
    }

    /// Get the number of stored objects
    pub fn len(&self) -> usize {
        self.objects.read().map(|o| o.len()).unwrap_or(0)
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all objects
    pub fn clear(&self) -> Result<()> {
        {
            let mut objects = self.objects.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            objects.clear();
        }

        {
            let mut type_idx = self.type_index.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            type_idx.clear();
        }

        {
            let mut topic_idx = self.topic_index.write().map_err(|e| {
                DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
            })?;
            topic_idx.clear();
        }

        if let Some(ref db) = self.db {
            db.clear()
                .map_err(|e| DpaError::ContextError(format!("Failed to clear database: {}", e)))?;
        }

        info!("Cleared all context objects");
        Ok(())
    }

    /// Load data from disk
    fn load_from_disk(&mut self) -> Result<()> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(()),
        };

        let mut count = 0;
        for result in db.iter() {
            let (_, value) = result
                .map_err(|e| DpaError::ContextError(format!("Failed to read from disk: {}", e)))?;

            let object: ContextObject = serde_json::from_slice(&value)?;

            // Don't use insert() to avoid writing back to disk
            let id = object.id;

            // Update indexes
            {
                let mut type_idx = self.type_index.write().map_err(|e| {
                    DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
                })?;
                type_idx
                    .entry(object.object_type)
                    .or_default()
                    .push(id);
            }

            {
                let mut topic_idx = self.topic_index.write().map_err(|e| {
                    DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
                })?;
                for topic in &object.topics {
                    topic_idx
                        .entry(topic.to_lowercase())
                        .or_default()
                        .push(id);
                }
            }

            // Add to vector store if embedding exists
            if let Some(ref emb) = object.embedding {
                let entry = VectorEntry {
                    id,
                    embedding: emb.clone(),
                    metadata: HashMap::new(),
                };
                self.vector_store.insert(entry)?;
            }

            // Add to main store
            {
                let mut objects = self.objects.write().map_err(|e| {
                    DpaError::ContextError(format!("Failed to acquire write lock: {}", e))
                })?;
                objects.insert(id, object);
            }

            count += 1;
        }

        info!("Loaded {} context objects from disk", count);
        Ok(())
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        if let Some(ref db) = self.db {
            db.flush()
                .map_err(|e| DpaError::ContextError(format!("Failed to flush: {}", e)))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for ContextStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextStore")
            .field("objects_count", &self.len())
            .field("has_persistence", &self.db.is_some())
            .finish()
    }
}
