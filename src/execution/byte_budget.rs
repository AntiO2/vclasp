use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Shared reservation for transient payload, packet and RGB working buffers.
/// Persistent caches, codec-private DPBs and caller-owned outputs are separate.
pub(crate) struct ByteBudget {
    capacity: usize,
    used: Mutex<usize>,
    changed: Condvar,
    peak: AtomicUsize,
    wait_ns: AtomicU64,
}

pub(crate) struct BytePermit {
    budget: Arc<ByteBudget>,
    bytes: usize,
}

impl ByteBudget {
    pub(crate) fn new(capacity: usize) -> Result<Arc<Self>, String> {
        if capacity == 0 {
            return Err("transient byte budget must be positive".into());
        }
        Ok(Arc::new(Self {
            capacity,
            used: Mutex::new(0),
            changed: Condvar::new(),
            peak: AtomicUsize::new(0),
            wait_ns: AtomicU64::new(0),
        }))
    }

    pub(crate) fn acquire(self: &Arc<Self>, bytes: usize) -> Result<BytePermit, String> {
        if bytes > self.capacity {
            return Err(format!("execution requires {bytes} transient bytes, exceeding budget {}; submit a smaller window or increase max_inflight_bytes", self.capacity));
        }
        let started = std::time::Instant::now();
        let mut waited = false;
        let mut used = self
            .used
            .lock()
            .map_err(|_| "transient byte budget poisoned")?;
        while bytes > self.capacity - *used {
            waited = true;
            used = self
                .changed
                .wait(used)
                .map_err(|_| "transient byte budget poisoned")?;
        }
        *used += bytes;
        self.peak.fetch_max(*used, Ordering::Relaxed);
        if waited {
            self.wait_ns.fetch_add(
                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
        })
    }

    pub(crate) fn snapshot(&self) -> (usize, usize, usize, u64) {
        let used = *self.used.lock().unwrap_or_else(|error| error.into_inner());
        (
            self.capacity,
            used,
            self.peak.load(Ordering::Relaxed),
            self.wait_ns.load(Ordering::Relaxed),
        )
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        let mut used = self
            .budget
            .used
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *used -= self.bytes;
        self.budget.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservations_are_shared_released_and_reject_oversize() {
        let budget = ByteBudget::new(10).unwrap();
        assert!(budget.acquire(11).is_err());
        let permit = budget.acquire(10).unwrap();
        let other = Arc::clone(&budget);
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _permit = other.acquire(1).unwrap();
            sender.send(()).unwrap();
        });
        assert!(receiver
            .recv_timeout(std::time::Duration::from_millis(20))
            .is_err());
        drop(permit);
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        worker.join().unwrap();
        assert_eq!(*budget.used.lock().unwrap(), 0);
    }
}
