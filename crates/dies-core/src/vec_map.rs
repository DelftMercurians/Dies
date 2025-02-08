use std::iter::FromIterator;

/// A map implementation backed by a vector of key-value pairs.
///
/// This collection is designed for small maps (approximately 10 elements or fewer) where
/// the keys don't necessarily implement `Hash`, or where the overhead of hash-based
/// lookup would be greater than linear search.
///
/// # When to use VecMap
///
/// - When you have a small number of elements (n ≈ 10 or less)
/// - When your keys don't implement `Hash`
/// - When you want predictable memory usage and cache locality
/// - When you want to avoid the overhead of hash computation
///
/// For larger collections, consider using `HashMap` or `BTreeMap` instead.
#[derive(Clone, Debug)]
pub struct VecMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> Default for VecMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K, V> VecMap<K, V> {
    /// Creates an empty `VecMap`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Creates an iterator over the key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }

    /// Creates a mutable iterator over the key-value pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (K, V)> {
        self.entries.iter_mut()
    }

    /// Returns an iterator over the keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over the values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Returns a mutable iterator over the values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.entries.iter_mut().map(|(_, v)| v)
    }
}

impl<K: PartialEq, V> VecMap<K, V> {
    /// Returns true if the map contains a value for the specified key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    /// If the map did have this key present, the value is updated, and the old value is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(index) = self.entries.iter().position(|(k, _)| k == &key) {
            let old = std::mem::replace(&mut self.entries[index], (key, value));
            Some(old.1)
        } else {
            self.entries.push((key, value));
            None
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    /// If the key doesn't exist, inserts the provided default value.
    pub fn get_or_insert_with<F>(&mut self, key: K, default: F) -> &mut V
    where
        F: FnOnce() -> V,
    {
        if let Some(index) = self.entries.iter().position(|(k, _)| k == &key) {
            &mut self.entries[index].1
        } else {
            self.entries.push((key, default()));
            &mut self.entries.last_mut().unwrap().1
        }
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(index) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.remove(index).1)
        } else {
            None
        }
    }
}

impl<K, V> FromIterator<(K, V)> for VecMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl<K, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut map = VecMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert("a", 1);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&"a"));
        assert_eq!(map.get(&"a"), Some(&1));

        assert_eq!(map.insert("a", 2), Some(1));
        assert_eq!(map.get(&"a"), Some(&2));

        assert_eq!(map.remove(&"a"), Some(2));
        assert!(map.is_empty());
    }

    #[test]
    fn test_iteration() {
        let mut map = VecMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec![&"a", &"b", &"c"]);

        let values: Vec<_> = map.values().collect();
        assert_eq!(values, vec![&1, &2, &3]);

        // Test FromIterator
        let collected: VecMap<_, _> = vec![("x", 10), ("y", 20)].into_iter().collect();
        assert_eq!(collected.get(&"x"), Some(&10));
        assert_eq!(collected.get(&"y"), Some(&20));
    }

    #[test]
    fn test_get_or_insert() {
        let mut map = VecMap::new();
        let value = map.get_or_insert_with("a", || 42);
        assert_eq!(*value, 42);

        *value = 100;
        assert_eq!(map.get(&"a"), Some(&100));
    }

    #[test]
    fn test_with_non_hash_type() {
        // Test with a type that doesn't implement Hash
        #[derive(Debug, PartialEq)]
        struct NonHashable(f64);

        let mut map = VecMap::new();
        map.insert(NonHashable(1.0), "one");
        map.insert(NonHashable(2.0), "two");

        assert_eq!(map.get(&NonHashable(1.0)), Some(&"one"));
        assert_eq!(map.get(&NonHashable(2.0)), Some(&"two"));
    }
}
