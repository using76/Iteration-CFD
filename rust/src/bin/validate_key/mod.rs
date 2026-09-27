// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The answer keys `ofgpu-validate` is held against: `reference/PROVENANCE.md`
//! names every published number a gate compares with - the file it lives in
//! (or the constant it still is), its source, DOI or URL, licence and SHA-256 -
//! and this module reads a key back through that manifest, refusing by name a
//! key whose file is absent or whose digest is not the manifest's. A gate then
//! reports its verdict as open rather than passing on nothing (SPEC-LIT §10,
//! the registry of §69).
//!
//! Written from: FIPS PUB 180-4, Secure Hash Standard, NIST (2015), section
//! 5.1.1 (padding), 5.3.3 (initial hash value), 6.2 (SHA-256 message schedule
//! and compression), 4.1.2 and 4.2.2 (the functions and constants) - a US
//! Government standard in the public domain; no implementation of it was
//! consulted. The manifest format is this repository's own.
//!
//! No GPL-licensed source was consulted.

// The manifest's provenance cells (`source`, `doi_url`, `licence`) are read only
// by the census test, and `ghia_1982` is called only by the ignored Ghia tests
// until K3 wires it into `run`; so the non-test binary leaves part of this
// module unused, by design. Reviewed when K3 lands.
#![cfg_attr(not(test), allow(dead_code))]

use std::path::{Path, PathBuf};

use ofgpu::{Error, Result};

/// The manifest's file name under [`reference_dir`].
pub const MANIFEST: &str = "PROVENANCE.md";

/// `<crate>/../reference`, located from the crate and never from the cwd
/// (`validate.rs`'s own `cases/` reads do the same; the JSON writer's git
/// probe uses `.` and is the wrong pattern for a file).
pub fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../reference")
}

/// `p` as text, with the separators a manifest and its messages name: `/`.
fn display_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

// ---------------------------------------------------------------- SHA-256 --
// FIPS PUB 180-4 (NIST, 2015): section 4.1.2's functions, section 4.2.2's
// constants, section 5.1.1's padding, section 5.3.3's initial hash value and
// section 6.2's message schedule and compression. Words are `u32`, every
// addition wraps, every rotate is `u32::rotate_right`.

/// The round constants of section 4.2.2, in order.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The initial hash value of section 5.3.3.
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// FIPS 180-4 SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = H0;
    // Section 5.1.1: append the byte 0x80, then zero bytes until the length
    // is 56 mod 64, then the message length IN BITS as a 64-bit big-endian
    // integer.
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.chunks_exact(64) {
        // Section 6.2: the message schedule. `w[t]` for `t < 16` is the
        // block's `t`-th big-endian word; for `16 <= t < 64`,
        // `w[t] = s1(w[t-2]) + w[t-7] + s0(w[t-15]) + w[t-16]`.
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([
                block[4 * t],
                block[4 * t + 1],
                block[4 * t + 2],
                block[4 * t + 3],
            ]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        // Section 6.2: the compression function over the working registers.
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].into_iter().enumerate() {
            h[i] = h[i].wrapping_add(v);
        }
    }

    // Output: `H[0..8]`, each as big-endian bytes.
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// [`sha256`] as 64 lower-case hex characters - what `sha256sum` prints.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for b in sha256(bytes) {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

// --------------------------------------------------------------- manifest --

/// Whether a key is a file under `reference/` or a literal still typed into a
/// source file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    File,
    Literal,
}

/// One row of the manifest's table, cells trimmed.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub kind: Kind,
    pub file: String,
    pub source: String,
    pub doi_url: String,
    pub licence: String,
    pub sha256: String,
}

/// The parsed table of `reference/PROVENANCE.md`.
#[derive(Debug)]
pub struct Manifest {
    pub path: PathBuf,
    pub rows: Vec<Row>,
}

impl Manifest {
    /// Read and parse `reference_dir()/PROVENANCE.md`.
    pub fn load() -> Result<Manifest> {
        let path = reference_dir().join(MANIFEST);
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
            path: display_path(&path),
            source,
        })?;
        Self::parse(&path, &text)
    }

    /// Parse manifest text. A line whose trimmed form starts with `|` is a
    /// table line: split on `|`, drop the empty first and last cells, trim
    /// each; skip the line whose first cell is `id` (the header) or starts
    /// with `-` (the rule). Everything else is prose and ignored. Refuses by
    /// name (`Error::Parse { path, msg }`, `msg` naming the line number and
    /// the id where there is one): a table line with other than 7 cells; a
    /// `kind` other than `file` / `literal`; a `file` row whose `sha256` is
    /// not exactly 64 characters of `[0-9a-f]`; a `literal` row whose
    /// `sha256` is not `-`; an `id` seen twice; an empty `id` or `file`.
    pub fn parse(path: &Path, text: &str) -> Result<Manifest> {
        let named = display_path(path);
        let mut rows: Vec<Row> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if !t.starts_with('|') {
                continue; // prose
            }
            let at = i + 1;
            let cells: Vec<&str> = t.split('|').map(str::trim).collect();
            // The line's own leading and trailing `|` produce the two empty
            // outer cells; the row is what lies between them.
            let cells = &cells[1..cells.len() - 1];
            if cells.len() != 7 {
                return Err(Error::Parse {
                    path: named.clone(),
                    msg: format!(
                        "line {at}: a table row needs 7 cells between its leading and \
                         trailing |, found {}",
                        cells.len()
                    ),
                });
            }
            if cells[0] == "id" || cells[0].starts_with('-') {
                continue; // the header row, then the rule under it
            }
            let kind = match cells[1] {
                "file" => Kind::File,
                "literal" => Kind::Literal,
                other => {
                    return Err(Error::Parse {
                        path: named.clone(),
                        msg: format!(
                            "line {at}: answer key \"{}\": the kind cell is \"{other}\", \
                             not file or literal",
                            cells[0]
                        ),
                    });
                }
            };
            let id = cells[0];
            if id.is_empty() {
                return Err(Error::Parse {
                    path: named.clone(),
                    msg: format!("line {at}: a row's id cell is empty"),
                });
            }
            if cells[2].is_empty() {
                return Err(Error::Parse {
                    path: named.clone(),
                    msg: format!("line {at}: answer key \"{id}\": the file cell is empty"),
                });
            }
            match kind {
                Kind::File => {
                    let sha = cells[6];
                    let lower_hex = sha.len() == 64
                        && sha
                            .bytes()
                            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
                    if !lower_hex {
                        return Err(Error::Parse {
                            path: named.clone(),
                            msg: format!(
                                "line {at}: answer key \"{id}\": a file row's sha256 must \
                                 be 64 lower-case hex characters, found \"{sha}\""
                            ),
                        });
                    }
                }
                Kind::Literal => {
                    if cells[6] != "-" {
                        return Err(Error::Parse {
                            path: named.clone(),
                            msg: format!(
                                "line {at}: answer key \"{id}\": a literal row's sha256 \
                                 cell is \"-\", found \"{}\"",
                                cells[6]
                            ),
                        });
                    }
                }
            }
            if rows.iter().any(|r| r.id == id) {
                return Err(Error::Parse {
                    path: named.clone(),
                    msg: format!("line {at}: answer key \"{id}\" names a second row"),
                });
            }
            rows.push(Row {
                id: id.to_string(),
                kind,
                file: cells[2].to_string(),
                source: cells[3].to_string(),
                doi_url: cells[4].to_string(),
                licence: cells[5].to_string(),
                sha256: cells[6].to_string(),
            });
        }
        Ok(Manifest {
            path: path.to_path_buf(),
            rows,
        })
    }

    /// The row whose `id` is `id`, if the manifest names one.
    pub fn row(&self, id: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.id == id)
    }
}

/// A key file, read and verified.
#[derive(Debug)]
pub struct KeyFile {
    pub id: String,
    /// The manifest's `file` cell (relative to `reference/`).
    pub file: String,
    pub path: PathBuf,
    /// The digest of the bytes read - equal to the manifest's, or `load` refused.
    pub sha256: String,
    /// The header row's names, in order.
    pub columns: Vec<String>,
    /// One `Vec<f64>` per data row, `columns.len()` long each.
    pub rows: Vec<Vec<f64>>,
}

impl KeyFile {
    /// The named column, or `Error::Parse` naming the column and the file.
    pub fn column(&self, name: &str) -> Result<Vec<f64>> {
        let at = self
            .columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| Error::Parse {
                path: display_path(&self.path),
                msg: format!(
                    "answer key \"{}\" ({}): no column named {name}",
                    self.id, self.file
                ),
            })?;
        Ok(self.rows.iter().map(|r| r[at]).collect())
    }

    /// The line a gate prints beside its verdict:
    /// `answer key <id>: reference/<file> sha256 <hex>` - exactly that shape.
    pub fn digest_line(&self) -> String {
        format!(
            "answer key {}: reference/{} sha256 {}",
            self.id, self.file, self.sha256
        )
    }
}

/// `load_from(&Manifest::load()?, &reference_dir(), id)`.
pub fn load(id: &str) -> Result<KeyFile> {
    load_from(&Manifest::load()?, &reference_dir(), id)
}

/// The verified half of a key read, shared by [`load_from`] and
/// [`load_text_from`]: the manifest's row, the joined path, the SHA-256 of
/// the bytes as read, and the bytes as UTF-8 text - with exactly the
/// refusals [`load_from`]'s doc comment names. One reader, one digest: the
/// two callers cannot disagree about what a key is.
fn read_verified<'a>(
    manifest: &'a Manifest,
    dir: &Path,
    id: &str,
) -> Result<(&'a Row, PathBuf, String, String)> {
    let named = display_path(&manifest.path);
    let row = manifest.row(id).ok_or_else(|| Error::Parse {
        path: named.clone(),
        msg: format!("no row of {named} names the answer key \"{id}\""),
    })?;
    if row.kind == Kind::Literal {
        return Err(Error::Parse {
            path: named,
            msg: format!(
                "answer key \"{id}\" is a literal in {}, not a file",
                row.file
            ),
        });
    }
    let path = dir.join(&row.file);
    let bytes = std::fs::read(&path).map_err(|source| Error::Io {
        path: display_path(&path),
        source,
    })?;
    let digest = sha256_hex(&bytes);
    if digest != row.sha256 {
        return Err(Error::Parse {
            path: display_path(&path),
            msg: format!(
                "sha256 {digest} does not match the manifest's {} for answer key \
                 \"{id}\" (line endings? the file is tracked with LF)",
                row.sha256
            ),
        });
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| Error::Parse {
        path: display_path(&path),
        msg: format!("answer key \"{id}\": the file is not UTF-8 text"),
    })?;
    Ok((row, path, digest, text.to_string()))
}

/// `load_text_from(&Manifest::load()?, &reference_dir(), id)`.
pub fn load_text(id: &str) -> Result<(String, String)> {
    load_text_from(&Manifest::load()?, &reference_dir(), id)
}

/// The key `id`'s digest line and VERIFIED text, for keys that are not the
/// CSV [`load_from`] parses - an MKM `.means` whitespace table with `%`
/// header lines, and the two transcriptions SPEC-LIT §110.3 and §110.4
/// print. The refusals are exactly [`load_from`]'s (no row names `id`; a
/// literal row; an unreadable file; a digest that is not the manifest's;
/// text that is not UTF-8) - one reader, one digest, so a key cannot verify
/// differently for the two callers. The returned line is
/// [`KeyFile::digest_line`]'s exact shape; making sense of the text is the
/// gate's own parser's job.
pub fn load_text_from(manifest: &Manifest, dir: &Path, id: &str) -> Result<(String, String)> {
    let (row, _path, digest, text) = read_verified(manifest, dir, id)?;
    let line = format!("answer key {}: reference/{} sha256 {}", id, row.file, digest);
    Ok((line, text))
}

/// The key `id` through `manifest`, its file under `dir`. Refuses by name
/// (`Error::Parse`, `msg` naming `id`): no row names `id`; the row is a
/// literal ("answer key `<id>` is a literal in <file>, not a file"); the
/// digest of the bytes is not the row's ("sha256 <got> does not match the
/// manifest's <want> for answer key `<id>` (line endings? the file is tracked
/// with LF)"); a data row whose cell count differs from the header's (line
/// number); a cell that does not parse as `f64` (line number, cell text).
/// An unreadable file is `Error::Io { path, source }` with `path` the joined
/// path, separators as `/`. CSV grammar: a line whose trimmed form is empty
/// or starts with `#` is skipped; the first remaining line is the header,
/// split on `,` and trimmed; every later line is data, split on `,`, each
/// cell `str::parse::<f64>` after trimming.
pub fn load_from(manifest: &Manifest, dir: &Path, id: &str) -> Result<KeyFile> {
    let (row, path, digest, text) = read_verified(manifest, dir, id)?;
    let mut columns: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if columns.is_empty() {
            columns = t.split(',').map(|c| c.trim().to_string()).collect();
            continue;
        }
        let at = i + 1;
        let cells: Vec<&str> = t.split(',').map(str::trim).collect();
        if cells.len() != columns.len() {
            return Err(Error::Parse {
                path: display_path(&path),
                msg: format!(
                    "line {at}: answer key \"{id}\": a data row has {} cells, \
                     the header has {}",
                    cells.len(),
                    columns.len()
                ),
            });
        }
        let mut values = Vec::with_capacity(cells.len());
        for cell in cells {
            values.push(cell.parse::<f64>().map_err(|_| Error::Parse {
                path: display_path(&path),
                msg: format!(
                    "line {at}: answer key \"{id}\": \"{cell}\" does not parse as a number"
                ),
            })?);
        }
        rows.push(values);
    }
    if columns.is_empty() {
        return Err(Error::Parse {
            path: display_path(&path),
            msg: format!("answer key \"{id}\": the file has no header row"),
        });
    }
    Ok(KeyFile {
        id: id.to_string(),
        file: row.file.clone(),
        path,
        sha256: digest,
        columns,
        rows,
    })
}

// ------------------------------------------------------------------ Ghia --

/// The empty skip list the Re = 100 comparison passes.
const NO_SKIP: &[usize] = &[];

/// Ghia, Ghia and Shin (1982), Tables I and II at Re = 100 and 400, from the
/// two key files. `v_re400_excluded` is the index list of stations flagged
/// `excluded_re400 = 1` (today exactly `[5]`, x = 0.9063).
pub struct Ghia {
    pub y: Vec<f64>,
    pub u_re100: Vec<f64>,
    pub u_re400: Vec<f64>,
    pub x: Vec<f64>,
    pub v_re100: Vec<f64>,
    pub v_re400: Vec<f64>,
    pub v_re400_excluded: Vec<usize>,
    /// The two key files, for their `digest_line`s.
    pub keys: [KeyFile; 2],
}

impl Ghia {
    /// `(u_table, v_table, v_skip)` for `re` in `{100, 400}`; `None` otherwise.
    pub fn columns(&self, re: u32) -> Option<(&[f64], &[f64], &[usize])> {
        match re {
            100 => Some((&self.u_re100, &self.v_re100, NO_SKIP)),
            400 => Some((&self.u_re400, &self.v_re400, &self.v_re400_excluded)),
            _ => None,
        }
    }
}

/// Loads `ghia1982-table-I` and `ghia1982-table-II`; refuses by name a table
/// that is not 17 stations, or whose header is not `y,u_re100,u_re400` /
/// `x,v_re100,v_re400,excluded_re400`.
pub fn ghia_1982() -> Result<Ghia> {
    // answer-key: ghia1982-table-I
    let t1 = load("ghia1982-table-I")?;
    // answer-key: ghia1982-table-II
    let t2 = load("ghia1982-table-II")?;
    const TABLE_I: &str = "y,u_re100,u_re400";
    const TABLE_II: &str = "x,v_re100,v_re400,excluded_re400";
    for (k, header) in [(&t1, TABLE_I), (&t2, TABLE_II)] {
        if k.columns.join(",") != header {
            return Err(Error::Parse {
                path: display_path(&k.path),
                msg: format!(
                    "answer key \"{}\": the header row is {}, expected {header}",
                    k.id,
                    k.columns.join(",")
                ),
            });
        }
        if k.rows.len() != 17 {
            return Err(Error::Parse {
                path: display_path(&k.path),
                msg: format!(
                    "answer key \"{}\": the table is {} stations, expected 17",
                    k.id,
                    k.rows.len()
                ),
            });
        }
    }
    Ok(Ghia {
        y: t1.column("y")?,
        u_re100: t1.column("u_re100")?,
        u_re400: t1.column("u_re400")?,
        x: t2.column("x")?,
        v_re100: t2.column("v_re100")?,
        v_re400: t2.column("v_re400")?,
        v_re400_excluded: t2
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r[3] == 1.0)
            .map(|(i, _)| i)
            .collect(),
        keys: [t1, t2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_reproduces_the_fips_180_4_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn every_file_key_matches_its_manifest_digest() {
        let m = Manifest::load().expect("reference/PROVENANCE.md parses");
        let files: Vec<&Row> = m.rows.iter().filter(|r| r.kind == Kind::File).collect();
        assert!(files.len() >= 2, "at least two file rows, found {}", files.len());
        for row in files {
            let bytes = std::fs::read(reference_dir().join(&row.file))
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", row.file));
            assert_eq!(sha256_hex(&bytes), row.sha256, "{} moved", row.file);
        }
    }

    #[test]
    fn ghia_1982_loads_seventeen_stations_and_the_erratum() {
        let g = ghia_1982().expect("the Ghia key loads");
        for v in [&g.y, &g.u_re100, &g.u_re400, &g.x, &g.v_re100, &g.v_re400] {
            assert_eq!(v.len(), 17, "every column is 17 stations");
        }
        assert_eq!(g.y[0], 1.0);
        assert_eq!(g.y[16], 0.0);
        assert_eq!(g.u_re100[1], 0.84123);
        assert_eq!(g.u_re400[10], -0.32726);
        assert_eq!(g.x[5], 0.9063);
        assert_eq!(g.v_re100[8], 0.05454);
        assert_eq!(g.v_re400[5], -0.23827);
        assert_eq!(g.v_re400_excluded, vec![5]);
        let (_, _, skip_100) = g.columns(100).expect("Re 100 is a table");
        assert!(skip_100.is_empty(), "the Re = 100 comparison skips nothing");
        let (_, _, skip_400) = g.columns(400).expect("Re 400 is a table");
        assert_eq!(skip_400.len(), 1);
        assert_eq!(skip_400[0], 5);
        assert!(g.columns(1000).is_none());
        assert_eq!(g.keys[0].column("u_re400").expect("the column")[10], -0.32726);
        for k in &g.keys {
            let line = k.digest_line();
            assert!(
                line.starts_with("answer key ghia1982-table-") && line.ends_with(&k.sha256),
                "{line}"
            );
            assert_eq!(k.sha256.len(), 64, "{line}");
        }
    }

    /// The header row and rule of the manifest's table, for the synthetic
    /// manifests below.
    fn synthetic_header() -> String {
        "| id | kind | file | source | DOI/URL | licence | sha256 |\n\
         |---|---|---|---|---|---|---|\n"
            .to_string()
    }

    /// The `Err` text of `r`, or a panic naming what came back instead.
    fn text_of<T: std::fmt::Debug>(r: Result<T>) -> String {
        match r {
            Ok(v) => panic!("expected a refusal, got {v:?}"),
            Err(e) => e.to_string(),
        }
    }

    /// A one-row manifest written beside `dir`, creating the directory.
    fn one_row_manifest(dir: &Path, row: &str) -> Manifest {
        std::fs::create_dir_all(dir).expect("the temp dir");
        let text = format!("{}{row}\n", synthetic_header());
        Manifest::parse(&dir.join("PROVENANCE.md"), &text).expect("the synthetic manifest parses")
    }

    const ZEROS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    #[test]
    fn a_missing_key_is_refused_naming_the_id() {
        let text = text_of(load("no-such-key"));
        assert!(text.contains("no-such-key"), "{text}");
    }

    #[test]
    fn load_text_returns_the_verified_text_of_a_whitespace_key() {
        let dir = std::env::temp_dir().join("ofgpuValidate_key_text");
        let body = "% Re_tau = 1\n 0 0\n";
        let m = one_row_manifest(
            &dir,
            &format!(
                "| k | file | k.means | s | u | l | {} |",
                sha256_hex(body.as_bytes())
            ),
        );
        std::fs::write(dir.join("k.means"), body).expect("write k.means");
        let (line, text) = load_text_from(&m, &dir, "k").expect("the whitespace key loads");
        assert_eq!(text, body);
        assert!(
            line.starts_with("answer key k: reference/k.means sha256 "),
            "{line}"
        );
        let other = "% Re_tau = 2\n 0 0\n";
        std::fs::write(dir.join("k.means"), other).expect("rewrite k.means");
        let got = text_of(load_text_from(&m, &dir, "k"));
        assert!(got.contains(&sha256_hex(other.as_bytes())), "{got}");
        assert!(got.contains(&sha256_hex(body.as_bytes())), "{got}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_digest_mismatch_is_refused_naming_both_digests() {
        let dir = std::env::temp_dir().join("ofgpuValidate_key_mismatch");
        let m = one_row_manifest(&dir, &format!("| k | file | k.csv | s | u | l | {ZEROS} |"));
        std::fs::write(dir.join("k.csv"), "a,b\n1,2\n").expect("write k.csv");
        let got = text_of(load_from(&m, &dir, "k"));
        assert!(got.contains(ZEROS), "{got}");
        assert!(got.contains(&sha256_hex(b"a,b\n1,2\n")), "{got}");
        assert!(got.contains('k'), "{got}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The three malformed-manifest refusals, as texts - shared with the
    /// verdict-word test below.
    fn malformed_texts() -> Vec<String> {
        let path = Path::new("PROVENANCE.md");
        let mut texts = Vec::new();
        for bad in [
            "| six | file | f.csv | s | u | l |",
            "| d | data | f.csv | s | u | l | - |",
            &format!(
                "| lit | literal | validate.rs X | s | u | l | {} |",
                "a".repeat(64)
            ),
        ] {
            texts.push(text_of(
                Manifest::parse(path, &format!("{}{bad}\n", synthetic_header())),
            ));
        }
        texts
    }

    #[test]
    fn a_malformed_manifest_row_is_refused_by_name() {
        let texts = malformed_texts();
        assert!(texts[0].contains(" 7 ") && texts[0].contains("line 3"), "{}", texts[0]);
        assert!(texts[1].contains("data") && texts[1].contains("\"d\""), "{}", texts[1]);
        assert!(texts[2].contains("lit"), "{}", texts[2]);
        // And a VALID literal row parses to one row of kind Literal.
        let text = format!(
            "{}| ok | literal | validate.rs Y | s | u | l | - |\n",
            synthetic_header()
        );
        let m = Manifest::parse(Path::new("PROVENANCE.md"), &text).expect("the valid row parses");
        assert_eq!(m.rows.len(), 1);
        assert_eq!(m.rows[0].kind, Kind::Literal);
    }

    #[test]
    fn no_error_text_spells_a_verdict_word() {
        // The four words, built from pieces so this file spells none of them
        // either - the run-time audit reads every printed line, and a test
        // that spells a word is a temptation.
        let words = [
            concat!("OP", "EN"),
            concat!("MIS", "SES"),
            concat!("MIS", "S"),
            concat!("MIS", "SED"),
        ];
        let shout = |text: &str| {
            text.split(|ch: char| !ch.is_ascii_alphabetic())
                .any(|w| words.contains(&w))
        };
        let mut texts: Vec<String> = vec![text_of(load("no-such-key"))];
        let dir = std::env::temp_dir().join("ofgpuValidate_key_words");
        let m = one_row_manifest(&dir, &format!("| k | file | k.csv | s | u | l | {ZEROS} |"));
        std::fs::write(dir.join("k.csv"), "a,b\n1,2\n").expect("write k.csv");
        texts.push(text_of(load_from(&m, &dir, "k")));
        texts.push(text_of(load_from(&m, &dir, "not-named")));
        texts.push(text_of(load_from(
            &one_row_manifest(&dir, "| lit | literal | somewhere | s | u | l | - |"),
            &dir,
            "lit",
        )));
        texts.extend(malformed_texts());
        for t in &texts {
            assert!(!shout(t), "an error text spells a verdict word: {t}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_answer_key_is_named_in_the_manifest_and_every_row_has_its_marker() {
        const SCAN: [(&str, &str); 5] = [
            ("rust/src/bin/validate.rs", include_str!("../validate.rs")),
            ("rust/src/bin/validate_key/mod.rs", include_str!("mod.rs")),
            ("rust/src/simple.rs", include_str!("../../simple.rs")),
            ("rust/src/fan/tests.rs", include_str!("../../fan/tests.rs")),
            ("rust/src/cht/ambient.rs", include_str!("../../cht/ambient.rs")),
        ];
        let needle: &str = concat!("// answer-key", ": ");
        let mut markers: Vec<(String, &str)> = Vec::new();
        for (file, text) in SCAN {
            for line in text.lines() {
                if let Some(id) = line.trim().strip_prefix(needle) {
                    markers.push((id.trim().to_string(), file));
                }
            }
        }
        let mut seen: Vec<&str> = markers.iter().map(|(id, _)| id.as_str()).collect();
        seen.sort_unstable();
        let distinct = seen.len();
        seen.dedup();
        assert_eq!(distinct, seen.len(), "a marker appears twice: {seen:?}");
        assert_eq!(markers.len(), 20, "twenty markers, found {markers:?}");
        let m = Manifest::load().expect("reference/PROVENANCE.md parses");
        let mut row_ids: Vec<&str> = m.rows.iter().map(|r| r.id.as_str()).collect();
        row_ids.sort_unstable();
        assert_eq!(seen, row_ids, "markers and rows must name the same ids");
        let file_ids: Vec<&str> = m
            .rows
            .iter()
            .filter(|r| r.kind == Kind::File)
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(
            file_ids,
            [
                "ghia1982-table-I",
                "ghia1982-table-II",
                "ho-powell-liley1972-silicon",
                "kadoya1985-table-7",
                "kadoya1985-table-11",
                "kadoya1985-tables-8-12",
                "nasa-glenn2002-coefficients",
                "nasa-glenn2002-table-B1",
            ]
        );
        for row in &m.rows {
            assert!(!row.source.is_empty(), "{}: an empty source cell", row.id);
            assert!(!row.doi_url.is_empty(), "{}: an empty DOI/URL cell", row.id);
            assert!(!row.licence.is_empty(), "{}: an empty licence cell", row.id);
            let kind = match row.kind {
                Kind::File => {
                    assert_eq!(row.sha256.len(), 64, "{}: a file row's sha256", row.id);
                    "file"
                }
                Kind::Literal => {
                    assert_eq!(row.sha256, "-", "{}: a literal row's sha256", row.id);
                    "literal"
                }
            };
            println!("  [answer-key] {}  {kind}  {}  {}", row.id, row.file, row.sha256);
        }
    }
}
