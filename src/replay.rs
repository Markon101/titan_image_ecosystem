//! Bounded replay scaffold. Not connected to the training recipe yet.
//! A caller MUST opt into ResetOnResume; no checkpoint sidecar is implied.
use crate::{state::WorldState, tensor_ops::splitmix64};
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayKind {
    Fresh,
    Mature,
    Damaged,
}
#[derive(Clone, Copy, Debug)]
pub enum PersistencePolicy {
    ResetOnResume,
}
pub struct Entry {
    pub source_fingerprint: u64,
    pub kind: ReplayKind,
    pub world: WorldState,
}
pub struct ReplayPool {
    entries: Vec<Entry>,
    capacity: usize,
    seed: u64,
    seen: u64,
    draws: u64,
    pub persistence: PersistencePolicy,
}
impl ReplayPool {
    pub fn new(capacity: usize, seed: u64, persistence: PersistencePolicy) -> Result<Self> {
        ensure!(
            (1..=16).contains(&capacity),
            "replay scaffold supports 1..=16 states"
        );
        Ok(Self {
            entries: Vec::new(),
            capacity,
            seed,
            seen: 0,
            draws: 0,
            persistence,
        })
    }
    pub fn insert(&mut self, source_fingerprint: u64, kind: ReplayKind, world: &WorldState) {
        self.seen += 1;
        let slot = if self.entries.len() < self.capacity {
            self.entries.len()
        } else {
            (splitmix64(self.seed ^ self.seen) % self.seen) as usize
        };
        if slot >= self.capacity {
            return;
        }
        let entry = Entry {
            source_fingerprint,
            kind,
            world: world.detached(),
        };
        if slot == self.entries.len() {
            self.entries.push(entry);
        } else {
            self.entries[slot] = entry;
        }
    }
    pub fn sample(&mut self, source_fingerprint: u64, kind: ReplayKind) -> Option<WorldState> {
        let matching: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.source_fingerprint == source_fingerprint && e.kind == kind)
            .collect();
        if matching.is_empty() {
            return None;
        }
        self.draws += 1;
        Some(
            matching[(splitmix64(self.seed ^ self.draws.rotate_left(32)) % matching.len() as u64)
                as usize]
                .world
                .detached(),
        )
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservoir_is_bounded_target_keyed_and_deterministic() -> Result<()> {
        let config = crate::RunConfig {
            micro_size: 8,
            macro_size: 4,
            channels: 12,
            interface_width: 8,
            ..Default::default()
        };
        let mut a = ReplayPool::new(3, 42, PersistencePolicy::ResetOnResume)?;
        let mut b = ReplayPool::new(3, 42, PersistencePolicy::ResetOnResume)?;
        for step in 0..40 {
            let mut world = WorldState::fresh(&config, step, &candle_core::Device::Cpu)?;
            world.step = step;
            a.insert(step % 2, ReplayKind::Mature, &world);
            b.insert(step % 2, ReplayKind::Mature, &world);
        }
        assert_eq!(a.len(), 3);
        assert!(a.sample(999, ReplayKind::Mature).is_none());
        assert_eq!(
            a.sample(0, ReplayKind::Mature).map(|w| w.step),
            b.sample(0, ReplayKind::Mature).map(|w| w.step)
        );
        assert!(a.sample(0, ReplayKind::Fresh).is_none());
        Ok(())
    }
}
