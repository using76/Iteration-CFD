// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The `identity` block of (92.57)'s run summary (SPEC-LIT §92.14): six
//! keys that name the mesh and the run that made it, so a summary on disk
//! can be keyed and joined. `mesh_id` is deterministic - FNV-1a 64 over
//! the case directory as configured plus the mesh name - because the mesh
//! at a directory is one mesh however many times it is re-made.
//!
//! Provenance: ORIGINAL - FNV-1a is Fowler/Noll/Vo's public-domain hash,
//! implemented from its published two-line definition and its published
//! test vectors; no implementation was read. The civil-date arithmetic is
//! the standard proleptic-Gregorian day-count inverse, textbook algebra,
//! written here because this crate takes no date dependency.
//! No GPL-licensed source was consulted.

/// The `identity` block's own version. Bump only when a key changes meaning.
pub const IDENTITY_SCHEMA: u32 = 1;

/// The mesh identity (92.57) records beside the numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshIdentity {
    pub mesh_id: String,
    pub run_id: Option<String>,
    pub tool: String,
    pub written_at: String,
    pub host: Option<String>,
}

/// FNV-1a, 64-bit, over the bytes as given.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `<case dir as configured, '\' -> '/', no trailing '/'>` + "\n" + `name` -
/// no absolutisation (CONTRACT §15): the id only has to be stable for one
/// writer writing one directory, which this is.
pub fn identity_key(case_dir: &std::path::Path, name: &str) -> String {
    let dir = case_dir.to_string_lossy().replace(['\\'], "/");
    let dir = dir.trim_end_matches('/');
    format!("{dir}\n{name}")
}

/// `"m_"` + 16 lowercase hex of `fnv1a64(identity_key(..).as_bytes())`.
pub fn mesh_id(case_dir: &std::path::Path, name: &str) -> String {
    format!("m_{:016x}", fnv1a64(identity_key(case_dir, name).as_bytes()))
}

/// `^[A-Za-z0-9._-]{1,64}$`, written as an explicit byte check (no regex
/// crate). Note `..` IS accepted: both of its characters are inside the
/// published set - the refusal of a path comes from the separator.
pub fn is_run_id(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 64
        && b.iter().all(|&c| {
            c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-'
        })
}

/// `OFGPU_RUN_ID` when it is a run id; `None` otherwise, having printed
/// C3's warning to stderr when the variable was set but malformed.
/// `tool` is the prefix of that warning line.
pub fn run_id_from_env(tool: &str) -> Option<String> {
    let Ok(v) = std::env::var("OFGPU_RUN_ID") else {
        return None;
    };
    if is_run_id(&v) {
        return Some(v);
    }
    eprintln!(
        "{tool}: OFGPU_RUN_ID='{v}' is not a run id (1 to 64 characters from \
         [A-Za-z0-9._-]); the summary records run_id: null"
    );
    None
}

/// C4's format, `YYYY-MM-DDTHH:MM:SSZ`, proleptic Gregorian, from the Unix
/// epoch second. Takes the instant so a test can pin it.
pub fn utc_stamp(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let s = (secs % 86_400) as u32;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        s / 3600,
        (s % 3600) / 60,
        s % 60
    )
}

/// The standard proleptic-Gregorian day-count inverse: a day number
/// (days since 1970-01-01) to `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `COMPUTERNAME`, else `HOSTNAME`, else `None`. An empty value is `None`.
pub fn host() -> Option<String> {
    for k in ["COMPUTERNAME", "HOSTNAME"] {
        if let Ok(v) = std::env::var(k) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

impl MeshIdentity {
    /// `written_at` is `utc_stamp(SystemTime::now())`; `host` is `host()`.
    pub fn new(
        tool: &str,
        case_dir: &std::path::Path,
        name: &str,
        run_id: Option<String>,
    ) -> Self {
        MeshIdentity {
            mesh_id: mesh_id(case_dir, name),
            run_id,
            tool: tool.to_string(),
            written_at: utc_stamp(std::time::SystemTime::now()),
            host: host(),
        }
    }

    /// C1's six keys. `run_id` and `host` serialise as JSON `null` when
    /// unknown - never a missing key, never the string `"null"`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": IDENTITY_SCHEMA,
            "mesh_id": self.mesh_id,
            "run_id": self.run_id,
            "tool": self.tool,
            "written_at": self.written_at,
            "host": self.host,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn fnv1a64_matches_the_published_vectors() {
        let hex = |b: &[u8]| format!("{:016x}", fnv1a64(b));
        assert_eq!(hex(b""), "cbf29ce484222325");
        assert_eq!(hex(b"a"), "af63dc4c8601ec8c");
        assert_eq!(hex(b"abc"), "e71fa2190541574b");
        assert_eq!(hex(b"foobar"), "85944171f73967e8");
    }

    #[test]
    fn mesh_id_is_the_directory_and_the_name() {
        let k = identity_key(Path::new("out/dir/"), "cube");
        assert!(!k.ends_with('/'), "{k:?}");
        assert_eq!(k.matches('\n').count(), 1, "{k:?}");
        let id = mesh_id(Path::new("out"), "cube");
        assert_eq!(id.len(), 18, "{id}");
        assert!(id.starts_with("m_"), "{id}");
        assert_eq!(mesh_id(Path::new("out"), "cube"), id);
        assert_ne!(mesh_id(Path::new("out"), "other"), id);
        assert_ne!(mesh_id(Path::new("out2"), "cube"), id);
        #[cfg(windows)]
        assert_eq!(
            mesh_id(Path::new("C:/out/cube"), "cube"),
            "m_ede67abc39071c48"
        );
    }

    #[test]
    fn utc_stamp_is_iso8601_utc() {
        assert_eq!(utc_stamp(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(
            utc_stamp(UNIX_EPOCH + Duration::from_secs(951_782_400)),
            "2000-02-29T00:00:00Z"
        );
        assert_eq!(
            utc_stamp(UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            "2023-11-14T22:13:20Z"
        );
        assert_eq!(
            utc_stamp(UNIX_EPOCH + Duration::from_secs(1_789_019_553)),
            "2026-09-10T05:52:33Z"
        );
    }

    #[test]
    fn is_run_id_accepts_the_minted_shape() {
        for ok in ["r_1", "r_221", "a", "A.b-c_9", "x".repeat(64).as_str()] {
            assert!(is_run_id(ok), "{ok:?} refused");
        }
        for bad in [
            "", "x".repeat(65).as_str(), "r 1", "../x", "a/b", r"a\b",
            "a:b", "a\"b", "a\nb",
        ] {
            assert!(!is_run_id(bad), "{bad:?} accepted");
        }
    }

    #[test]
    fn identity_to_json_has_six_keys() {
        let ident = MeshIdentity {
            mesh_id: "m_ede67abc39071c48".to_string(),
            run_id: None,
            tool: "ofgpu-automesher".to_string(),
            written_at: "2026-09-15T04:12:33Z".to_string(),
            host: Some("DESKTOP".to_string()),
        };
        let v = ident.to_json();
        let obj = v.as_object().expect("an object");
        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["host", "mesh_id", "run_id", "schema", "tool", "written_at"]);
        assert!(v["schema"].is_number(), "schema is a number, not a string");
        assert_eq!(v["schema"], 1);
        assert_eq!(v["run_id"], serde_json::Value::Null);
    }
}
