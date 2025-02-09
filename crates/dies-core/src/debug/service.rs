use arc_swap::ArcSwap;
use dashmap::DashSet;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};

use super::types::DebugValue;

/// Update event for debug values
#[derive(Clone)]
pub enum DebugUpdate {
    /// Insert or update a debug value
    Insert { key: String, value: DebugValue },
    /// Clear a debug value
    Clear { key: String },
}

/// Handle for subscribing to debug value updates
pub struct DebugSubscriber {
    rx: mpsc::UnboundedReceiver<DebugUpdate>,
    service_tx: mpsc::UnboundedSender<ServiceMsg>,
}

impl DebugSubscriber {
    /// Receives the next debug update
    pub async fn recv(&mut self) -> Option<DebugUpdate> {
        self.rx.recv().await
    }

    /// Sets whether a debug key is active (being monitored)
    pub fn set_active(&self, key: &str, active: bool) {
        self.service_tx.send(ServiceMsg::SetActive {
            key: key.to_string(),
            active,
        });
    }

    /// Gets a list of all available debug keys
    pub async fn get_available_keys(&self) -> Vec<String> {
        let (sender, receiver) = oneshot::channel();
        self.service_tx
            .send(ServiceMsg::GetAvailableKeys(sender))
            .unwrap();
        receiver.await.unwrap()
    }

    /// Gets a copy of all current debug values
    pub async fn get_copy(&self) -> HashMap<String, DebugValue> {
        let (sender, receiver) = oneshot::channel();
        self.service_tx.send(ServiceMsg::GetCopy(sender)).unwrap();
        receiver.await.unwrap()
    }
}

enum ServiceMsg {
    SetActive { key: String, active: bool },
    GetAvailableKeys(oneshot::Sender<Vec<String>>),
    GetCopy(oneshot::Sender<HashMap<String, DebugValue>>),
    Send(DebugUpdate),
}

/// Service for managing debug values with pub/sub functionality
#[derive(Clone)]
pub struct DebugService {
    active_prefixes: Arc<ArcSwap<Arc<BTreeSet<String>>>>,
    available_keys: Arc<DashSet<String>>,
    tx: mpsc::UnboundedSender<DebugUpdate>,
}

impl DebugService {
    /// Spawns a new debug service and returns a subscriber
    pub fn spawn() -> (DebugService, DebugSubscriber) {
        let active_prefixes =
            Arc::new(ArcSwap::from(Arc::new(Arc::new(BTreeSet::<String>::new()))));
        let available_keys = Arc::new(DashSet::<String>::new());
        let (tx, rx) = mpsc::unbounded_channel();

        let (service_tx, mut service_rx) = mpsc::unbounded_channel();
        {
            let available_keys = available_keys.clone();
            let active_prefixes = active_prefixes.clone();
            tokio::spawn(async move {
                let mut value_map = HashMap::new();
                while let Some(msg) = service_rx.recv().await {
                    match msg {
                        ServiceMsg::SetActive { key, active } => {
                            let mut new_prefixes = (*active_prefixes.load_full()).as_ref().clone();
                            if active {
                                new_prefixes.insert(key);
                            } else {
                                new_prefixes.remove(&key);
                            }
                            active_prefixes.store(Arc::new(Arc::new(new_prefixes)));
                        }
                        ServiceMsg::GetAvailableKeys(sender) => {
                            sender
                                .send(available_keys.iter().map(|k| k.clone()).collect())
                                .unwrap();
                        }
                        ServiceMsg::GetCopy(sender) => {
                            let _ = sender.send(value_map.clone());
                        }
                        ServiceMsg::Send(update) => match update {
                            DebugUpdate::Insert { key, value } => {
                                value_map.insert(key, value);
                            }
                            DebugUpdate::Clear { key } => {
                                value_map.remove(&key);
                            }
                        },
                    }
                }
            });
        };

        let service = Self {
            active_prefixes,
            available_keys,
            tx,
        };
        let subscriber = DebugSubscriber { rx, service_tx };
        (service, subscriber)
    }

    /// Sends a debug value update for a given key
    pub fn send(&self, key: &str, value: Option<DebugValue>) {
        // Make sure key is in available keys
        if !self.available_keys.contains(key) {
            self.available_keys.insert(key.to_string());
        }
        // Check if key is active
        if is_prefix(self.active_prefixes.load_full().as_ref(), key) {
            match value.as_ref() {
                Some(value) => {
                    let _ = self.tx.send(DebugUpdate::Insert {
                        key: key.to_string(),
                        value: value.clone(),
                    });
                }
                None => {
                    let _ = self.tx.send(DebugUpdate::Clear {
                        key: key.to_string(),
                    });
                }
            }
        }
    }
}

fn is_prefix(prefixes: &BTreeSet<String>, key: &str) -> bool {
    // Find the first prefix that could potentially match (using BTreeSet's ordering)
    if let Some(candidate) = prefixes.range(..=key.to_string()).next_back() {
        return key.starts_with(candidate);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prefix() {
        let mut prefixes = BTreeSet::new();
        prefixes.insert("robot".to_string());
        prefixes.insert("world".to_string());
        prefixes.insert("team/blue".to_string());

        // Test exact matches
        assert!(is_prefix(&prefixes, "robot"));
        assert!(is_prefix(&prefixes, "world"));
        assert!(is_prefix(&prefixes, "team/blue"));

        // Test prefix matches
        assert!(is_prefix(&prefixes, "robot/position"));
        assert!(is_prefix(&prefixes, "world/ball"));
        assert!(is_prefix(&prefixes, "team/blue/robot1"));

        // Test non-matches
        assert!(!is_prefix(&prefixes, "robo"));
        assert!(!is_prefix(&prefixes, "worl"));
        assert!(!is_prefix(&prefixes, "team"));
        assert!(!is_prefix(&prefixes, "team/red"));
        assert!(!is_prefix(&prefixes, ""));

        // Test with empty prefix set
        let empty_prefixes = BTreeSet::new();
        assert!(!is_prefix(&empty_prefixes, "anything"));
        assert!(!is_prefix(&empty_prefixes, ""));
    }

    #[test]
    fn test_is_prefix_ordering() {
        let mut prefixes = BTreeSet::new();
        prefixes.insert("a/b".to_string());
        prefixes.insert("a".to_string());

        // Both "a" and "a/b" are valid prefixes for "a/b/c"
        // The function should match "a" as it's the shorter prefix
        assert!(is_prefix(&prefixes, "a/b/c"));
        assert!(is_prefix(&prefixes, "a/other"));
    }
}
