//! A bounded map with a Python `dict`'s insertion order, for the server's per-game caches.

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

/// A map that remembers insertion order like a Python `dict`: updating a key keeps its place,
/// and [`OrderedMap::pop_first`] removes the oldest key (`del d[next(iter(d))]`).
#[derive(Debug)]
pub struct OrderedMap<K, V> {
    map: HashMap<K, (u64, V)>,
    order: BTreeMap<u64, K>,
    next: u64,
}

impl<K: Hash + Eq + Clone, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        OrderedMap {
            map: HashMap::new(),
            order: BTreeMap::new(),
            next: 0,
        }
    }
}

impl<K: Hash + Eq + Clone, V> OrderedMap<K, V> {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The value of `key`.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|(_, v)| v)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// `d[key] = value`: a new key goes last, an existing one keeps its place.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(slot) = self.map.get_mut(&key) {
            slot.1 = value;
            return;
        }
        let seq = self.next;
        self.next += 1;
        self.order.insert(seq, key.clone());
        self.map.insert(key, (seq, value));
    }

    /// `d.pop(key, None)`.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (seq, v) = self.map.remove(key)?;
        self.order.remove(&seq);
        Some(v)
    }

    /// Removes and returns the oldest entry.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        let (_, key) = self.order.pop_first()?;
        let (_, v) = self.map.remove(&key)?;
        Some((key, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_python_dict_order() {
        let mut m = OrderedMap::new();
        m.insert("a", 1);
        m.insert("b", 2);
        m.insert("a", 3); // keeps its place
        m.insert("c", 4);
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&"a"), Some(&3));
        assert_eq!(m.pop_first(), Some(("a", 3)));
        assert_eq!(m.remove(&"c"), Some(4));
        assert_eq!(m.remove(&"c"), None);
        m.insert("a", 5); // goes last now
        assert_eq!(m.pop_first(), Some(("b", 2)));
        assert_eq!(m.pop_first(), Some(("a", 5)));
        assert!(m.is_empty() && m.pop_first().is_none());
    }
}
