use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairCandidate {
    pub sample_id: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationSelection {
    pub sample_ids: Vec<u64>,
    pub used_bytes: u64,
    pub budget_bytes: u64,
    pub calibration_requests: usize,
    pub calibration_unique: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairSourceDescriptor {
    pub sample_id: u64,
    pub video_id: u64,
    pub anchor_offset: u64,
    pub anchor_length: u64,
    pub delta_offset: u64,
    pub delta_length: u64,
    pub target_ordinal: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializedPairDescriptor {
    pub sample_id: u64,
    pub video_id: u64,
    pub offset: u64,
    pub length: u64,
    pub target_ordinal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairMaterialization {
    pub descriptors: Vec<MaterializedPairDescriptor>,
    pub bytes: u64,
}

fn copy_extent(
    source: &mut File,
    output: &mut BufWriter<File>,
    offset: u64,
    length: u64,
) -> Result<(), String> {
    source
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek source extent {offset}:{length}: {error}"))?;
    let copied = io::copy(&mut source.take(length), output)
        .map_err(|error| format!("copy source extent {offset}:{length}: {error}"))?;
    if copied != length {
        return Err(format!(
            "source extent {offset}:{length} ended after {copied} bytes"
        ));
    }
    Ok(())
}

fn temporary_path(output: &Path) -> PathBuf {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{}", std::process::id()));
    output.with_file_name(name)
}

pub fn materialize_pairs(
    source_path: &Path,
    output_path: &Path,
    descriptors: &[PairSourceDescriptor],
    selected_sample_ids: Option<&[u64]>,
) -> Result<PairMaterialization, String> {
    if source_path == output_path {
        return Err("Pair source and output paths must differ".to_string());
    }
    if output_path.exists() {
        return Err(format!(
            "Pair output already exists: {}",
            output_path.display()
        ));
    }
    let source_bytes = source_path
        .metadata()
        .map_err(|error| format!("stat Pair source {}: {error}", source_path.display()))?
        .len();
    let mut by_id = HashMap::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if descriptor.anchor_length == 0 {
            return Err(format!(
                "sample {} has an empty Anchor",
                descriptor.sample_id
            ));
        }
        if descriptor.target_ordinal == 0 && descriptor.delta_length != 0 {
            return Err(format!(
                "Anchor target {} unexpectedly has Delta bytes",
                descriptor.sample_id
            ));
        }
        if descriptor.target_ordinal > 0 && descriptor.delta_length == 0 {
            return Err(format!(
                "Delta target {} has no Delta bytes",
                descriptor.sample_id
            ));
        }
        for (name, offset, length) in [
            ("Anchor", descriptor.anchor_offset, descriptor.anchor_length),
            ("Delta", descriptor.delta_offset, descriptor.delta_length),
        ] {
            let end = offset.checked_add(length).ok_or_else(|| {
                format!("sample {} {name} extent overflows", descriptor.sample_id)
            })?;
            if end > source_bytes {
                return Err(format!(
                    "sample {} {name} extent {offset}:{length} exceeds source bytes {source_bytes}",
                    descriptor.sample_id
                ));
            }
        }
        if by_id.insert(descriptor.sample_id, *descriptor).is_some() {
            return Err(format!(
                "duplicate Pair source sample {}",
                descriptor.sample_id
            ));
        }
    }

    let selected = match selected_sample_ids {
        Some(sample_ids) => {
            let set = sample_ids.iter().copied().collect::<HashSet<_>>();
            if set.len() != sample_ids.len() {
                return Err("Pair materialization selection contains duplicates".to_string());
            }
            for sample_id in &set {
                if !by_id.contains_key(sample_id) {
                    return Err(format!("selected unknown Pair sample {sample_id}"));
                }
            }
            set
        }
        None => by_id.keys().copied().collect(),
    };
    let mut ordered = selected
        .into_iter()
        .map(|sample_id| by_id[&sample_id])
        .collect::<Vec<_>>();
    ordered.sort_by_key(|descriptor| descriptor.sample_id);

    let temporary = temporary_path(output_path);
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|error| {
            format!(
                "remove stale Pair temporary {}: {error}",
                temporary.display()
            )
        })?;
    }
    let result = (|| {
        let mut source = File::open(source_path)
            .map_err(|error| format!("open Pair source {}: {error}", source_path.display()))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("create Pair temporary {}: {error}", temporary.display()))?;
        let mut output = BufWriter::new(file);
        let mut output_offset = 0u64;
        let mut materialized = Vec::with_capacity(ordered.len());
        for descriptor in ordered {
            copy_extent(
                &mut source,
                &mut output,
                descriptor.anchor_offset,
                descriptor.anchor_length,
            )?;
            if descriptor.delta_length > 0 {
                copy_extent(
                    &mut source,
                    &mut output,
                    descriptor.delta_offset,
                    descriptor.delta_length,
                )?;
            }
            let length = descriptor
                .anchor_length
                .checked_add(descriptor.delta_length)
                .ok_or_else(|| format!("sample {} Pair length overflows", descriptor.sample_id))?;
            materialized.push(MaterializedPairDescriptor {
                sample_id: descriptor.sample_id,
                video_id: descriptor.video_id,
                offset: output_offset,
                length,
                target_ordinal: usize::from(descriptor.target_ordinal > 0),
            });
            output_offset = output_offset
                .checked_add(length)
                .ok_or_else(|| "Pair output length overflows".to_string())?;
        }
        output
            .flush()
            .map_err(|error| format!("flush Pair temporary {}: {error}", temporary.display()))?;
        drop(output);
        fs::rename(&temporary, output_path).map_err(|error| {
            format!(
                "publish Pair output {} -> {}: {error}",
                temporary.display(),
                output_path.display()
            )
        })?;
        Ok(PairMaterialization {
            descriptors: materialized,
            bytes: output_offset,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn select_profiled_pairs(
    candidates: &[PairCandidate],
    calibration_trace: &[u64],
    budget_bytes: u64,
) -> Result<MaterializationSelection, String> {
    let mut sizes = HashMap::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.bytes == 0 {
            return Err(format!(
                "sample {} has zero pair bytes",
                candidate.sample_id
            ));
        }
        if sizes.insert(candidate.sample_id, candidate.bytes).is_some() {
            return Err(format!(
                "duplicate candidate sample {}",
                candidate.sample_id
            ));
        }
    }
    let mut counts = HashMap::<u64, u64>::new();
    for &sample_id in calibration_trace {
        if !sizes.contains_key(&sample_id) {
            return Err(format!("calibration references unknown sample {sample_id}"));
        }
        *counts.entry(sample_id).or_default() += 1;
    }
    let calibration_unique = counts.len();
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left_count), (right_id, right_count)| {
        let left_bytes = sizes[left_id];
        let right_bytes = sizes[right_id];
        let left_density = (*left_count as u128) * (right_bytes as u128);
        let right_density = (*right_count as u128) * (left_bytes as u128);
        right_density
            .cmp(&left_density)
            .then_with(|| right_count.cmp(left_count))
            .then_with(|| left_bytes.cmp(&right_bytes))
            .then_with(|| left_id.cmp(right_id))
    });

    let mut sample_ids = Vec::new();
    let mut used_bytes = 0u64;
    for (sample_id, _) in ranked {
        let bytes = sizes[&sample_id];
        if used_bytes
            .checked_add(bytes)
            .is_some_and(|next| next <= budget_bytes)
        {
            used_bytes += bytes;
            sample_ids.push(sample_id);
        }
    }
    debug_assert!(used_bytes <= budget_bytes);
    debug_assert_eq!(
        sample_ids.iter().copied().collect::<HashSet<_>>().len(),
        sample_ids.len()
    );
    Ok(MaterializationSelection {
        sample_ids,
        used_bytes,
        budget_bytes,
        calibration_requests: calibration_trace.len(),
        calibration_unique,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("vclasp-{name}-{}-{nonce}", std::process::id()))
    }

    fn descriptor(sample_id: u64, anchor_offset: u64, delta_offset: u64) -> PairSourceDescriptor {
        PairSourceDescriptor {
            sample_id,
            video_id: 9,
            anchor_offset,
            anchor_length: 2,
            delta_offset,
            delta_length: if delta_offset == 0 { 0 } else { 2 },
            target_ordinal: if delta_offset == 0 { 0 } else { 1 },
        }
    }

    #[test]
    fn selection_is_deterministic_and_never_exceeds_budget() {
        let candidates = [
            PairCandidate {
                sample_id: 1,
                bytes: 60,
            },
            PairCandidate {
                sample_id: 2,
                bytes: 40,
            },
            PairCandidate {
                sample_id: 3,
                bytes: 30,
            },
        ];
        let trace = [1, 1, 2, 2, 2, 3];
        let selected = select_profiled_pairs(&candidates, &trace, 70).unwrap();
        assert_eq!(selected.sample_ids, vec![2, 3]);
        assert_eq!(selected.used_bytes, 70);
        assert_eq!(selected.calibration_requests, 6);
        assert_eq!(selected.calibration_unique, 3);
    }

    #[test]
    fn zero_budget_selects_nothing() {
        let selected = select_profiled_pairs(
            &[PairCandidate {
                sample_id: 7,
                bytes: 1,
            }],
            &[7, 7],
            0,
        )
        .unwrap();
        assert!(selected.sample_ids.is_empty());
        assert_eq!(selected.used_bytes, 0);
    }

    #[test]
    fn evaluation_ids_cannot_enter_without_calibration_occurrence() {
        let candidates = [
            PairCandidate {
                sample_id: 1,
                bytes: 10,
            },
            PairCandidate {
                sample_id: 2,
                bytes: 10,
            },
        ];
        let selected = select_profiled_pairs(&candidates, &[1], 20).unwrap();
        assert_eq!(selected.sample_ids, vec![1]);
    }

    #[test]
    fn unknown_calibration_sample_is_rejected() {
        let error = select_profiled_pairs(
            &[PairCandidate {
                sample_id: 1,
                bytes: 10,
            }],
            &[2],
            10,
        )
        .unwrap_err();
        assert!(error.contains("unknown sample 2"));
    }

    #[test]
    fn pair_materialization_copies_ground_truth_extents_in_sample_order() {
        let source = test_path("pair-source");
        let output = test_path("pair-output");
        fs::write(&source, b"ABCDEFGHIJ").unwrap();
        let mut distant_target = descriptor(1, 0, 5);
        distant_target.target_ordinal = 7;
        let result = materialize_pairs(
            &source,
            &output,
            &[descriptor(2, 2, 0), distant_target],
            None,
        )
        .unwrap();

        assert_eq!(fs::read(&output).unwrap(), b"ABFGCD");
        assert_eq!(result.bytes, 6);
        assert_eq!(result.descriptors[0].sample_id, 1);
        assert_eq!(result.descriptors[0].offset, 0);
        assert_eq!(result.descriptors[0].length, 4);
        assert_eq!(result.descriptors[0].target_ordinal, 1);
        assert_eq!(result.descriptors[1].sample_id, 2);
        assert_eq!(result.descriptors[1].offset, 4);
        assert_eq!(result.descriptors[1].length, 2);
        assert_eq!(result.descriptors[1].target_ordinal, 0);
        fs::remove_file(source).unwrap();
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn pair_materialization_uses_only_selected_ids() {
        let source = test_path("pair-selected-source");
        let output = test_path("pair-selected-output");
        fs::write(&source, b"ABCDEFGHIJ").unwrap();
        let result = materialize_pairs(
            &source,
            &output,
            &[descriptor(1, 0, 5), descriptor(2, 2, 0)],
            Some(&[2]),
        )
        .unwrap();

        assert_eq!(fs::read(&output).unwrap(), b"CD");
        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].sample_id, 2);
        fs::remove_file(source).unwrap();
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn pair_materialization_rejects_out_of_bounds_without_publishing() {
        let source = test_path("pair-invalid-source");
        let output = test_path("pair-invalid-output");
        fs::write(&source, b"AB").unwrap();
        let error = materialize_pairs(&source, &output, &[descriptor(1, 0, 5)], None).unwrap_err();

        assert!(error.contains("exceeds source bytes"));
        assert!(!output.exists());
        fs::remove_file(source).unwrap();
    }
}
