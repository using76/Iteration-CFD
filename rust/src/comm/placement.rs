// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Which device each rank of an SPMD run uses (SPEC-LIT §138.7).
//!
//! Rank `r` goes to `list[r mod list.len()]`. The list is every device the
//! driver sees, `0..device_count`, unless `OFGPU_DEVICES` names the ordinals
//! to use, in order, comma separated. So four ranks on a one-device machine
//! all go to device 0, four ranks on a two-device machine go to 0, 1, 0, 1,
//! and `OFGPU_DEVICES=1` puts every rank on device 1.
//!
//! This is a pure function of its inputs, so it is tested with faked device
//! counts. The device count comes from the caller
//! ([`Gpu::device_count`](crate::device::Gpu::device_count)), and
//! [`Gpu::for_rank`](crate::device::Gpu::for_rank) opens the rank's device.
//! This module is host only, like the rest of
//! [`crate::comm`], and both builds compile and run the same code.
//!
//! Provenance: ORIGINAL - the round-robin placement and the override list are
//! this project's own design (docs/17 PAR-05). No external source. No
//! GPL-licensed source was consulted.

use crate::error::{Error, Result};

/// The environment variable that overrides the placement: a comma-separated
/// list of device ordinals, such as `0,1`. Entries are trimmed, and an
/// ordinal may appear more than once.
pub const DEVICES_ENV: &str = "OFGPU_DEVICES";

/// The devices ranks are placed on, cycled in rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    devices: Vec<usize>,
    device_count: usize,
    overridden: bool,
}

fn refuse(msg: String) -> Error {
    Error::Config(format!("placement: {msg}"))
}

impl Placement {
    /// The placement over `device_count` devices, or over the ordinals in
    /// `devices` (the value of [`DEVICES_ENV`]) when it is `Some`.
    ///
    /// Refused by name: no device at all; a list that is empty or blank; an
    /// empty entry; an entry that is not a device ordinal; an ordinal the
    /// device count does not have.
    pub fn new(device_count: usize, devices: Option<&str>) -> Result<Self> {
        if device_count == 0 {
            return Err(refuse(
                "no CUDA device to place a rank on (the device count is 0)".to_string(),
            ));
        }
        let Some(value) = devices else {
            return Ok(Self {
                devices: (0..device_count).collect(),
                device_count,
                overridden: false,
            });
        };
        if value.trim().is_empty() {
            return Err(refuse(format!(
                "{DEVICES_ENV} is set but empty; unset it to place rank r on device \
                 r mod {device_count}, or list device ordinals such as 0,1"
            )));
        }
        let mut list = Vec::new();
        for (i, entry) in value.split(',').enumerate() {
            let e = entry.trim();
            if e.is_empty() {
                return Err(refuse(format!(
                    "{DEVICES_ENV} entry {} is empty (value '{value}')",
                    i + 1
                )));
            }
            let d: usize = e.parse().map_err(|_| {
                refuse(format!(
                    "{DEVICES_ENV} entry {} '{e}' is not a device ordinal (value '{value}')",
                    i + 1
                ))
            })?;
            if d >= device_count {
                return Err(refuse(format!(
                    "{DEVICES_ENV} entry {} is device {d}, but there are {device_count} \
                     device(s), ordinals 0..{} (value '{value}')",
                    i + 1,
                    device_count - 1
                )));
            }
            list.push(d);
        }
        Ok(Self {
            devices: list,
            device_count,
            overridden: true,
        })
    }

    /// [`Placement::new`] with the override read from [`DEVICES_ENV`]; an
    /// unset variable means no override. A value that is not UTF-8 is refused
    /// by name.
    pub fn from_env(device_count: usize) -> Result<Self> {
        match std::env::var_os(DEVICES_ENV) {
            None => Self::new(device_count, None),
            Some(v) => match v.to_str() {
                Some(s) => Self::new(device_count, Some(s)),
                None => Err(refuse(format!("{DEVICES_ENV} is not valid UTF-8"))),
            },
        }
    }

    /// The device ordinal rank `rank` uses: `devices()[rank mod len]`.
    pub fn device_of(&self, rank: usize) -> usize {
        self.devices[rank % self.devices.len()]
    }

    /// The ordinals ranks cycle over, in order.
    pub fn devices(&self) -> &[usize] {
        &self.devices
    }

    /// The device count the placement was built for.
    pub fn device_count(&self) -> usize {
        self.device_count
    }

    /// True when the list came from [`DEVICES_ENV`].
    pub fn is_overridden(&self) -> bool {
        self.overridden
    }
}
