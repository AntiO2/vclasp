use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

/// Reuse signal computed from registered codec closures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCandidate {
    pub visible_consumers: usize,
    /// Later targets in the registered dependency group that could be served
    /// by monotonic decoder progress, even when they are not in this window.
    pub potential_consumers: usize,
    pub next_use_batch: Option<usize>,
}

/// Runtime liveness derived from the closure consumers in a bounded request
/// window. This state is intentionally not persisted in the on-disk index:
/// the same AU has different remaining consumers for every request window.
#[derive(Debug, Clone, Default)]
pub struct WindowDependencyLiveness {
    consumers_by_batch: Vec<HashMap<u64, usize>>,
}

impl WindowDependencyLiveness {
    pub fn from_batches(batches: Vec<HashMap<u64, usize>>) -> Self {
        Self {
            consumers_by_batch: batches,
        }
    }

    pub fn batch_consumers(&self, record_id: u64, batch: usize) -> usize {
        self.consumers_by_batch
            .get(batch)
            .and_then(|consumers| consumers.get(&record_id))
            .copied()
            .unwrap_or(0)
    }

    /// Reuses visible from `batch`: another consumer in the current batch or
    /// any consumer in a later batch. The first current-batch consumer pays
    /// for producing the dependency and is therefore not itself a reuse.
    pub fn visible_reuses(&self, record_id: u64, batch: usize) -> usize {
        let current = self.batch_consumers(record_id, batch);
        let future = self
            .consumers_by_batch
            .iter()
            .skip(batch.saturating_add(1))
            .map(|consumers| consumers.get(&record_id).copied().unwrap_or(0))
            .sum::<usize>();
        current.saturating_sub(1) + future
    }
}

/// Replacement policies manage checkpoint admission and eviction without
/// depending on decoder internals.
pub trait ResidentStatePolicy<K>: Send {
    fn admit(&self, candidate: ResidentCandidate) -> bool;
    fn on_insert(&mut self, key: K);
    fn on_hit(&mut self, key: &K);
    fn set_priority(&mut self, key: &K, pinned: bool, next_use_batch: Option<usize>);
    fn residency_counts(&self) -> (usize, usize);
    fn on_remove(&mut self, key: &K);
    fn victim(&mut self) -> Option<K>;
}

#[derive(Debug, Clone, Copy, Default)]
struct ResidencyPriority {
    pinned: bool,
    next_use_batch: Option<usize>,
}

/// Admit states with visible reuse or registered forward consumers. Only
/// visible reuse pins a state; unknown future reuse remains probationary and
/// is evicted before pinned state.
pub struct DependencyLivenessLru<K> {
    order: VecDeque<K>,
    resident: HashSet<K>,
    priorities: HashMap<K, ResidencyPriority>,
}

impl<K> DependencyLivenessLru<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            order: VecDeque::new(),
            resident: HashSet::new(),
            priorities: HashMap::new(),
        }
    }

    fn touch(&mut self, key: &K) {
        if let Some(position) = self.order.iter().position(|candidate| candidate == key) {
            self.order.remove(position);
        }
        self.order.push_back(key.clone());
    }
}

impl<K> ResidentStatePolicy<K> for DependencyLivenessLru<K>
where
    K: Clone + Eq + Hash + Send,
{
    fn admit(&self, candidate: ResidentCandidate) -> bool {
        (candidate.visible_consumers > 0 && candidate.next_use_batch.is_some())
            || candidate.potential_consumers > 0
    }

    fn on_insert(&mut self, key: K) {
        if self.resident.insert(key.clone()) {
            self.priorities
                .insert(key.clone(), ResidencyPriority::default());
            self.order.push_back(key);
        }
    }

    fn on_hit(&mut self, key: &K) {
        if self.resident.contains(key) {
            self.touch(key);
        }
    }

    fn set_priority(&mut self, key: &K, pinned: bool, next_use_batch: Option<usize>) {
        if let Some(priority) = self.priorities.get_mut(key) {
            *priority = ResidencyPriority {
                pinned,
                next_use_batch,
            };
        }
    }

    fn on_remove(&mut self, key: &K) {
        self.resident.remove(key);
        self.priorities.remove(key);
    }

    fn residency_counts(&self) -> (usize, usize) {
        let pinned = self
            .resident
            .iter()
            .filter(|key| {
                self.priorities
                    .get(*key)
                    .is_some_and(|priority| priority.pinned)
            })
            .count();
        (pinned, self.resident.len().saturating_sub(pinned))
    }

    fn victim(&mut self) -> Option<K> {
        let probationary = self.order.iter().position(|key| {
            self.resident.contains(key)
                && !self
                    .priorities
                    .get(key)
                    .is_some_and(|priority| priority.pinned)
        });
        let position = probationary.or_else(|| {
            self.order
                .iter()
                .enumerate()
                .filter(|(_, key)| self.resident.contains(*key))
                .max_by_key(|(_, key)| {
                    self.priorities
                        .get(*key)
                        .and_then(|priority| priority.next_use_batch)
                        .unwrap_or(usize::MAX)
                })
                .map(|(position, _)| position)
        })?;
        let key = self.order.remove(position)?;
        if self.resident.remove(&key) {
            self.priorities.remove(&key);
            Some(key)
        } else {
            self.victim()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DependencyLivenessLru, ResidentCandidate, ResidentStatePolicy, WindowDependencyLiveness,
    };
    use std::collections::HashMap;

    #[test]
    fn liveness_policy_rejects_state_without_visible_reuse() {
        let policy = DependencyLivenessLru::<u64>::new();
        assert!(!policy.admit(ResidentCandidate {
            visible_consumers: 0,
            potential_consumers: 0,
            next_use_batch: None,
        }));
        assert!(policy.admit(ResidentCandidate {
            visible_consumers: 0,
            potential_consumers: 1,
            next_use_batch: None,
        }));
        assert!(!policy.admit(ResidentCandidate {
            visible_consumers: 1,
            potential_consumers: 0,
            next_use_batch: None,
        }));
        assert!(policy.admit(ResidentCandidate {
            visible_consumers: 1,
            potential_consumers: 0,
            next_use_batch: Some(1),
        }));
    }

    #[test]
    fn liveness_lru_promotes_hits_and_evicts_probationary_first() {
        let mut policy = DependencyLivenessLru::new();
        policy.on_insert(1);
        policy.on_insert(2);
        policy.on_hit(&1);
        policy.set_priority(&1, true, Some(1));
        assert_eq!(policy.victim(), Some(2));
        assert_eq!(policy.victim(), Some(1));
    }

    #[test]
    fn removed_entry_is_not_considered_for_lru_eviction() {
        let mut policy = DependencyLivenessLru::new();
        policy.on_insert(1);
        policy.on_insert(2);
        policy.on_remove(&1);
        assert_eq!(policy.victim(), Some(2));
        assert_eq!(policy.victim(), None);
    }

    #[test]
    fn window_liveness_exposes_only_consumers_after_the_current_request() {
        let liveness = WindowDependencyLiveness::from_batches(vec![
            HashMap::from([(10, 1), (11, 1)]),
            HashMap::from([(10, 1)]),
        ]);
        assert_eq!(liveness.visible_reuses(10, 0), 1);
        assert_eq!(liveness.visible_reuses(10, 1), 0);
        assert_eq!(liveness.visible_reuses(11, 0), 0);
    }
}
