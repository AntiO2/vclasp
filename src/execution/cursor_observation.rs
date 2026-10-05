use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CursorActivity {
    pub live_cursors: usize,
    pub detached_cursors: usize,
    pub prefetched_payload_bytes: usize,
    pub prefetched_payload_capacity_bytes: usize,
    pub created_cursors: u64,
    pub destroyed_cursors: u64,
    pub sequence: u64,
}

pub(crate) type CursorObservation = Arc<Mutex<CursorActivity>>;

pub(crate) struct CursorLease {
    observation: CursorObservation,
    detached: bool,
    payload: usize,
    capacity: usize,
}

impl CursorLease {
    pub fn new(observation: CursorObservation) -> Self {
        {
            let mut state = observation.lock().unwrap_or_else(|e| e.into_inner());
            state.live_cursors += 1;
            state.created_cursors += 1;
            state.sequence += 1;
        }
        Self {
            observation,
            detached: false,
            payload: 0,
            capacity: 0,
        }
    }

    pub fn set_detached(&mut self, detached: bool) {
        if self.detached == detached {
            return;
        }
        let mut state = self.observation.lock().unwrap_or_else(|e| e.into_inner());
        if detached {
            state.detached_cursors += 1;
        } else {
            state.detached_cursors -= 1;
        }
        self.detached = detached;
        state.sequence += 1;
    }

    pub fn add_prefetch(&mut self, payload: usize, capacity: usize) {
        let mut state = self.observation.lock().unwrap_or_else(|e| e.into_inner());
        self.payload += payload;
        self.capacity += capacity;
        state.prefetched_payload_bytes += payload;
        state.prefetched_payload_capacity_bytes += capacity;
        state.sequence += 1;
    }

    pub fn remove_prefetch(&mut self, payload: usize, capacity: usize) {
        let mut state = self.observation.lock().unwrap_or_else(|e| e.into_inner());
        self.payload -= payload;
        self.capacity -= capacity;
        state.prefetched_payload_bytes -= payload;
        state.prefetched_payload_capacity_bytes -= capacity;
        state.sequence += 1;
    }
}

impl Drop for CursorLease {
    fn drop(&mut self) {
        let mut state = self.observation.lock().unwrap_or_else(|e| e.into_inner());
        state.live_cursors -= 1;
        if self.detached {
            state.detached_cursors -= 1;
        }
        state.prefetched_payload_bytes -= self.payload;
        state.prefetched_payload_capacity_bytes -= self.capacity;
        state.destroyed_cursors += 1;
        state.sequence += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detached_worker_and_error_drop_preserve_live_accounting() {
        let observation = CursorObservation::default();
        let mut lease = CursorLease::new(observation.clone());
        lease.add_prefetch(8, 64);
        lease.set_detached(true);
        let reader = observation.clone();
        std::thread::spawn(move || {
            let state = reader.lock().unwrap();
            assert_eq!(state.live_cursors, 1);
            assert_eq!(state.detached_cursors, 1);
            assert_eq!(state.prefetched_payload_capacity_bytes, 64);
        })
        .join()
        .unwrap();
        lease.remove_prefetch(8, 64);
        lease.add_prefetch(4, 32);
        lease.set_detached(false);
        lease.set_detached(true);
        drop(lease);
        let state = observation.lock().unwrap();
        assert_eq!(state.live_cursors, 0);
        assert_eq!(state.detached_cursors, 0);
        assert_eq!(state.prefetched_payload_bytes, 0);
        assert_eq!(state.prefetched_payload_capacity_bytes, 0);
        assert_eq!(state.created_cursors, state.destroyed_cursors);
    }
}
