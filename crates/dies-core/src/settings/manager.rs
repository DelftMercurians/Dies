use anyhow::Result;
use dashmap::DashMap;
use serde_json::Value;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    time::Duration,
};
use tokio::{sync::mpsc, time};

use super::Settings;

#[derive(Clone)]
pub struct SettingsHandle {
    inner: SettingsHandleInner,
}

#[derive(Clone)]
enum SettingsHandleInner {
    Fs(FsStorageHandle),
    InMemory(InMemoryStore),
}

impl SettingsHandle {
    pub fn open_in_memory() -> Self {
        Self {
            inner: SettingsHandleInner::InMemory(InMemoryStore::new()),
        }
    }

    pub fn open_directory(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: SettingsHandleInner::Fs(FsStorageHandle::open_directory(path)?),
        })
    }

    pub fn get<T: Settings + 'static>(&self) -> Arc<T> {
        match &self.inner {
            SettingsHandleInner::Fs(fs) => fs.get(),
            SettingsHandleInner::InMemory(im) => im.get(),
        }
    }

    pub fn save<T: Settings + 'static>(&self, value: T) {
        match &self.inner {
            SettingsHandleInner::Fs(fs) => fs.save(value),
            SettingsHandleInner::InMemory(im) => im.save(value),
        }
    }
}

#[derive(Clone)]
struct FsStorageHandle {
    unparsed: Arc<DashMap<String, serde_json::Value>>,
    parsed: Arc<DashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
    save_queue: mpsc::UnboundedSender<(String, Value)>,
}

impl FsStorageHandle {
    fn open_directory(path: &Path) -> Result<Self> {
        // Read all json files in the directory -> file name stem is the settings name
        let unparsed = Arc::new(DashMap::new());
        for entry in path.read_dir()? {
            let entry = entry?;
            if entry.path().is_file() && entry.path().extension().map_or(false, |ext| ext == "json")
            {
                let file = std::fs::File::open(entry.path())?;
                let settings_loaded: Value = serde_json::from_reader(&file)?;
                let file_stem = entry
                    .path()
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                unparsed.insert(file_stem, settings_loaded);
            }
        }
        let (tx, mut rx) = mpsc::unbounded_channel();
        let path = path.to_path_buf();
        tokio::spawn(async move {
            let mut pending_saves: HashMap<String, Value> = HashMap::new();
            let mut save_timer = time::interval(Duration::from_millis(200));
            let mut open_files: HashMap<String, BufWriter<File>> = HashMap::new();

            loop {
                tokio::select! {
                    Some((name, value)) = rx.recv() => {
                        pending_saves.insert(name, value);
                    }
                    _ = save_timer.tick() => {
                        if !pending_saves.is_empty() {
                            let saves = std::mem::take(&mut pending_saves);
                            for (name, value) in saves {
                                let writer = open_files.entry(name.clone()).or_insert_with(|| {
                                    let file = File::create(path.join(format!("{}.json", name)))
                                        .expect("Failed to create/open settings file");
                                    BufWriter::new(file)
                                });

                                // Seek to start and truncate
                                if let Ok(()) = writer.get_mut().seek(SeekFrom::Start(0)).map(|_| ()) {
                                    if let Ok(()) = writer.get_mut().set_len(0) {
                                        let _ = serde_json::to_writer(&mut *writer, &value);
                                        let _ = writer.flush();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        Ok(Self {
            unparsed,
            parsed: Arc::new(DashMap::new()),
            save_queue: tx,
        })
    }

    fn get<T: Settings + 'static>(&self) -> Arc<T> {
        // Try from parsed first
        let type_id = TypeId::of::<T>();
        if let Some(v) = self
            .parsed
            .get(&type_id)
            .and_then(|v| Arc::downcast(v.clone()).ok())
        {
            v
        } else if let Some(unparsed) = self
            .unparsed
            .get(T::name())
            .and_then(|v| serde_json::from_value::<T>(v.clone()).ok())
        {
            // Store in parsed
            let v = Arc::new(unparsed);
            self.parsed.insert(type_id, v.clone());
            v
        } else {
            Arc::new(T::default())
        }
    }

    fn save<T: Settings + 'static>(&self, v: T) {
        let type_id = TypeId::of::<T>();
        let v_json = serde_json::to_value(&v).expect("Failed to serialize settings");
        self.parsed.insert(type_id, Arc::new(v));
        let _ = self.save_queue.send((T::name().to_string(), v_json));
    }
}

#[derive(Clone, Default)]
struct InMemoryStore {
    parsed: Arc<DashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl InMemoryStore {
    fn new() -> Self {
        Self::default()
    }
}

impl InMemoryStore {
    fn get<T: Settings + 'static>(&self) -> Arc<T> {
        let type_id = TypeId::of::<T>();
        if let Some(v) = self
            .parsed
            .get(&type_id)
            .and_then(|v| Arc::downcast(v.clone()).ok())
        {
            v
        } else {
            Arc::new(T::default())
        }
    }

    fn save<T: Settings + 'static>(&self, value: T) {
        let type_id = TypeId::of::<T>();
        self.parsed.insert(type_id, Arc::new(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::descriptor::{FieldDesc, StructDesc, TypeDesc, ValueDesc};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestSettings {
        value: i32,
        name: String,
    }

    impl Settings for TestSettings {
        fn name() -> &'static str {
            "test_settings"
        }

        fn descriptor() -> TypeDesc {
            TypeDesc::Struct(StructDesc {
                type_name: "TestSettings".to_string(),
                label: "Test Settings".to_string(),
                description: Some("Settings used for testing".to_string()),
                fields: vec![
                    FieldDesc {
                        name: "value".to_string(),
                        label: "Value".to_string(),
                        description: Some("A test integer value".to_string()),
                        field_type: TypeDesc::Value(ValueDesc::Int {
                            min: None,
                            max: None,
                            step: None,
                        }),
                    },
                    FieldDesc {
                        name: "name".to_string(),
                        label: "Name".to_string(),
                        description: Some("A test string value".to_string()),
                        field_type: TypeDesc::Value(ValueDesc::String { pattern: None }),
                    },
                ],
            })
        }
    }

    impl Default for TestSettings {
        fn default() -> Self {
            Self {
                value: 42,
                name: "default".to_string(),
            }
        }
    }

    #[test]
    fn test_in_memory_store() {
        let store = InMemoryStore::new();

        // Test default value
        let settings = store.get::<TestSettings>();
        assert_eq!(settings.value, 42);
        assert_eq!(settings.name, "default");

        // Test saving and retrieving value
        let new_settings = TestSettings {
            value: 100,
            name: "test".to_string(),
        };
        store.save(new_settings.clone());

        let retrieved = store.get::<TestSettings>();
        assert_eq!(*retrieved, new_settings);
    }

    #[tokio::test]
    async fn test_fs_store() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let store = FsStorageHandle::open_directory(temp_dir.path()).unwrap();

        // Test default value
        let settings = store.get::<TestSettings>();
        assert_eq!(settings.value, 42);
        assert_eq!(settings.name, "default");

        // Test saving and retrieving value
        let new_settings = TestSettings {
            value: 100,
            name: "test".to_string(),
        };
        store.save(new_settings.clone());

        // Give some time for the async save to complete
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Create a new store instance to read from disk
        let store2 = FsStorageHandle::open_directory(temp_dir.path()).unwrap();
        let retrieved = store2.get::<TestSettings>();
        assert_eq!(*retrieved, new_settings);
    }
}
