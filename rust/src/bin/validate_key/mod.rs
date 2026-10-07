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
        assert_eq!(markers.len(), 27, "twenty-seven markers, found {markers:?}");
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
                "mcbride-gordon-reno1993-table-II",
                "ivptestset2008-rober",
                "ivptestset2008-hires",
                "tnf-flameD-centreline",
                "tnf-flameD-x15",
                "tnf-flameD-x30",
                "tnf-flameD-x45",
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

    // ------------------------------------------------- chemistry answer keys --

    /// The gas constant the TM-4513 coefficients were generated with,
    /// J/(mol K) (SPEC-LIT section 128.5; Gate 100-B uses the same value).
    const TM4513_R: f64 = 8.314510;

    /// cp/R, TM-4513 eq. (1) = SPEC-LIT eq. (128.1), `a = [a1..a7]` of one row.
    fn tm4513_cp_r(a: &[f64; 7], t: f64) -> f64 {
        let t2 = t * t;
        a[0] + a[1] * t + a[2] * t2 + a[3] * t2 * t + a[4] * t2 * t2
    }

    /// H/(R T), TM-4513 eq. (2) = SPEC-LIT eq. (128.2).
    fn tm4513_h_rt(a: &[f64; 7], t: f64) -> f64 {
        let t2 = t * t;
        a[0] + a[1] * t / 2.0 + a[2] * t2 / 3.0 + a[3] * t2 * t / 4.0 + a[4] * t2 * t2 / 5.0
            + a[5] / t
    }

    /// S/R, TM-4513 eq. (3) = SPEC-LIT eq. (128.3).
    fn tm4513_s_r(a: &[f64; 7], t: f64) -> f64 {
        let t2 = t * t;
        a[0] * t.ln() + a[1] * t + a[2] * t2 / 2.0 + a[3] * t2 * t / 3.0 + a[4] * t2 * t2 / 4.0
            + a[6]
    }

    /// The element counts of a TM-4513 species code, as (count, weight) pairs
    /// over SPEC-LIT section 128.2's weights, g/mol.
    fn tm4513_elements(code: i64) -> &'static [(f64, f64)] {
        const H: f64 = 1.00794;
        const C: f64 = 12.011;
        const N: f64 = 14.00674;
        const O: f64 = 15.9994;
        match code {
            1 => &[(2.0, N)],
            2 => &[(2.0, O)],
            5 => &[(1.0, C), (4.0, H)],
            6 => &[(1.0, C), (2.0, O)],
            7 => &[(2.0, H), (1.0, O)],
            8 => &[(1.0, C), (1.0, O)],
            9 => &[(2.0, H)],
            10 => &[(1.0, O), (1.0, H)],
            11 => &[(1.0, O)],
            12 => &[(1.0, H)],
            _ => unreachable!("no TM-4513 species code {code}"),
        }
    }

    #[test]
    fn tm4513_records_meet_at_1000_k_and_reproduce_their_printed_weight_and_h298() {
        // answer-key: mcbride-gordon-reno1993-table-II
        const ID: &str = "mcbride-gordon-reno1993-table-II";
        let key = load(ID).expect("the TM-4513 Table II key loads");
        // The row indices below follow this header.
        assert_eq!(
            key.columns,
            [
                "species", "t_lo", "t_hi", "mol_weight", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                "h298_r",
            ],
            "{ID}: the header row"
        );
        assert_eq!(key.rows.len(), 20, "{ID}: ten species, two rows each");
        let col = |name: &str| {
            key.column(name)
                .unwrap_or_else(|e| panic!("{ID}: the column {name}: {e}"))
        };
        let species = col("species");
        let t_lo = col("t_lo");
        let t_hi = col("t_hi");
        let mol_weight = col("mol_weight");
        let h298_r = col("h298_r");
        // [a1..a7] of each row, columns 4..=10 of the asserted header.
        let coeffs: Vec<[f64; 7]> = key
            .rows
            .iter()
            .map(|r| [r[4], r[5], r[6], r[7], r[8], r[9], r[10]])
            .collect();
        assert_eq!(
            species,
            [
                1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0,
                11.0, 11.0, 12.0, 12.0,
            ],
            "{ID}: the species column, two rows per code"
        );
        for (k, code) in species.iter().step_by(2).enumerate() {
            let (u, l) = (2 * k, 2 * k + 1);
            assert_eq!(
                (t_lo[u], t_hi[u]),
                (1000.0, 6000.0),
                "{ID}: row {u} (species {code}) is the 1000-6000 K row"
            );
            assert_eq!(
                (t_lo[l], t_hi[l]),
                (200.0, 1000.0),
                "{ID}: row {l} (species {code}) is the 200-1000 K row"
            );
            assert_eq!(
                mol_weight[u], mol_weight[l],
                "{ID}: species {code}: mol_weight agrees on rows {u} and {l}"
            );
            assert_eq!(
                h298_r[u], h298_r[l],
                "{ID}: species {code}: h298_r agrees on rows {u} and {l}"
            );
            // The fit at T_mid = 1000 K: the two rows' eqs. (1)-(3) agree
            // there to 1e-7 relative (worst measured 2.54e-8, CH4 H/RT).
            let at_mid = [
                (
                    tm4513_cp_r(&coeffs[u], 1000.0),
                    tm4513_cp_r(&coeffs[l], 1000.0),
                ),
                (
                    tm4513_h_rt(&coeffs[u], 1000.0),
                    tm4513_h_rt(&coeffs[l], 1000.0),
                ),
                (
                    tm4513_s_r(&coeffs[u], 1000.0),
                    tm4513_s_r(&coeffs[l], 1000.0),
                ),
            ];
            let mut jump = [0.0; 3];
            for (j, &(fu, fl)) in at_mid.iter().enumerate() {
                jump[j] = (fu - fl).abs() / fu.abs().max(fl.abs()).max(1.0);
                assert!(
                    jump[j] <= 1.0e-7,
                    "{ID}: species {code} rows {u}/{l}: eq. ({}) is {:e} apart at 1000 K, over 1e-7",
                    j + 1,
                    jump[j]
                );
            }
            // The printed H(298.15)/R, from the LOWER row's coefficients
            // (worst measured 6.40e-5, O).
            let h298 = 298.15 * tm4513_h_rt(&coeffs[l], 298.15);
            let dh = (h298 - h298_r[l]).abs();
            assert!(
                dh <= 1.0e-4,
                "{ID}: species {code} rows {u}/{l}: H(298.15)/R evaluates to {h298}, printed {}",
                h298_r[l]
            );
            // The printed molecular weight, from section 128.2's weights
            // (half the printed fifth decimal; worst measured 3.6e-15).
            let weight: f64 = tm4513_elements(*code as i64)
                .iter()
                .map(|(n, w)| n * w)
                .sum();
            let dw = (weight - mol_weight[l]).abs();
            assert!(
                dw <= 5.0e-6,
                "{ID}: species {code} rows {u}/{l}: the weights sum to {weight}, printed {}",
                mol_weight[l]
            );
            println!(
                "  [answer-key] tm4513  species {}  fit 1000 K {:e}  H(298.15)/R {:e}  weight {:e}",
                code,
                jump[0].max(jump[1]).max(jump[2]),
                dh,
                dw
            );
        }
        // Exact transcriptions, the spot rows a retyping would bend first.
        assert_eq!(coeffs[4][0], 1.63552643, "{ID}: row 4 (CH4, 1000-6000 K) a1");
        assert_eq!(coeffs[15][5], 3615.08056, "{ID}: row 15 (OH, 200-1000 K) a6");
        assert_eq!(coeffs[19][0], 2.5, "{ID}: row 19 (H, 200-1000 K) a1");
        for (j, a) in coeffs[19].iter().enumerate().skip(1).take(4) {
            assert_eq!(*a, 0.0, "{ID}: row 19 (H, 200-1000 K) a{}", j + 1);
        }
        assert_eq!(h298_r[0], 0.0, "{ID}: row 0 (N2) h298_r");
        assert_eq!(h298_r[17], 29968.7009, "{ID}: row 17 (O, 200-1000 K) h298_r");
    }

    #[test]
    fn table_b1_holds_twelve_species_and_meets_the_tm4513_records() {
        const ID: &str = "nasa-glenn2002-table-B1";
        const TM: &str = "mcbride-gordon-reno1993-table-II";
        let b1 = load(ID).expect("the Table B1 key loads");
        let tm = load(TM).expect("the TM-4513 Table II key loads");
        assert_eq!(
            b1.columns,
            ["species", "mol_weight", "cp_298", "dfh_298"],
            "{ID}: the header row"
        );
        assert_eq!(b1.rows.len(), 12, "{ID}: twelve species");
        let col = |k: &KeyFile, name: &str| {
            k.column(name)
                .unwrap_or_else(|e| panic!("{}: the column {name}: {e}", k.id))
        };
        let species = col(&b1, "species");
        let cp_298 = col(&b1, "cp_298");
        let dfh_298 = col(&b1, "dfh_298");
        assert_eq!(
            species,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "{ID}: the species column, in code order"
        );
        assert_eq!(cp_298[0], 29.124, "{ID}: row 0 (N2) cp_298, as Gate 100-B reads it");
        assert_eq!(cp_298[2], 20.786, "{ID}: row 2 (Ar) cp_298");
        assert_eq!(dfh_298[3], -0.126, "{ID}: row 3 (Air) dfh_298");
        assert_eq!(b1.rows[4][1], 16.04246, "{ID}: row 4 (CH4) mol_weight");
        assert_eq!(dfh_298[9], 37.278, "{ID}: row 9 (OH) dfh_298");
        assert_eq!(dfh_298[11], 217.999, "{ID}: row 11 (H) dfh_298");

        let tm_species = col(&tm, "species");
        let tm_h298_r = col(&tm, "h298_r");
        // [a1..a7] of each TM-4513 row, columns 4..=10 of its header.
        let coeffs: Vec<[f64; 7]> = tm
            .rows
            .iter()
            .map(|r| [r[4], r[5], r[6], r[7], r[8], r[9], r[10]])
            .collect();
        for (k, code) in tm_species.iter().step_by(2).enumerate() {
            let l = 2 * k + 1;
            let at = species
                .iter()
                .position(|s| s == code)
                .unwrap_or_else(|| panic!("{ID}: no row names species {code}"));
            // Cp(298.15) from the 200-1000 K row against the B1 column: the
            // 0.1 % is the fit difference between the seven- and
            // nine-coefficient forms (SPEC-LIT section 128.7; worst measured
            // 1.95e-5, O).
            let cp = TM4513_R * tm4513_cp_r(&coeffs[l], 298.15);
            let dcp = (cp - cp_298[at]).abs() / cp_298[at];
            assert!(
                dcp <= 1.0e-3,
                "{TM}: species {code} against {ID} row {at}: cp(298.15) is {cp} against {} (rel {dcp:e})",
                cp_298[at]
            );
            // Delta_f H(298.15): the row's H(298.15)/R in K, times R, in
            // kJ/mol, against the B1 column.
            let d = TM4513_R * tm_h298_r[l] / 1000.0 - dfh_298[at];
            if *code as i64 == 10 {
                // TM-4513's OH record (TPIS78) and TP-2002's Table B1 adopt
                // different heats of formation of OH, 39.347106 against
                // 37.278 kJ/mol. The difference is a fact of the two printed
                // tables, pinned here so nobody meets it unannounced.
                assert!(
                    (d - 2.069106).abs() <= 1.0e-5,
                    "{TM}: species 10 (OH) against {ID} row {at}: the heat-of-formation gap is {d} kJ/mol, pinned 2.069106"
                );
            } else {
                assert!(
                    d.abs() <= 0.010,
                    "{TM}: species {code} against {ID} row {at}: the heat of formation differs by {d} kJ/mol"
                );
            }
            println!(
                "  [answer-key] b1  species {}  cp {:.6}  cp_298 {}  rel {:e}  d {:+.6}",
                code, cp, cp_298[at], dcp, d
            );
        }
    }

    #[test]
    fn ivp_testset_references_hold_their_invariants() {
        // answer-key: ivptestset2008-rober
        // answer-key: ivptestset2008-hires
        const ROBER: &str = "ivptestset2008-rober";
        const HIRES: &str = "ivptestset2008-hires";
        let rober = load(ROBER).expect("the ROBER reference loads");
        let hires = load(HIRES).expect("the HIRES reference loads");
        for (k, id) in [(&rober, ROBER), (&hires, HIRES)] {
            assert_eq!(
                k.columns,
                ["component", "t_end", "y0", "y_ref"],
                "{id}: the header row"
            );
        }

        // ROBER, Table II.10.3 at t = 1e11.
        assert_eq!(rober.rows.len(), 3, "{ROBER}: three components");
        let rcol = |name: &str| {
            rober.column(name)
                .unwrap_or_else(|e| panic!("{ROBER}: the column {name}: {e}"))
        };
        let component = rcol("component");
        let t_end = rcol("t_end");
        let y0 = rcol("y0");
        let y_ref = rcol("y_ref");
        assert_eq!(component, [1.0, 2.0, 3.0], "{ROBER}: the component column");
        for (i, t) in t_end.iter().enumerate() {
            assert_eq!(*t, 1.0e11, "{ROBER}: row {i}: t_end");
        }
        assert_eq!(y0, [1.0, 0.0, 0.0], "{ROBER}: the y0 column, y(0) = (1, 0, 0)");
        assert_eq!(y_ref[0], 2.083340149701255e-8, "{ROBER}: row 0: y_ref");
        assert_eq!(y_ref[1], 8.333360770334713e-14, "{ROBER}: row 1: y_ref");
        assert_eq!(y_ref[2], 0.999999979166505, "{ROBER}: row 2: y_ref");
        // ROBER conserves mass, y1 + y2 + y3 = 1. The reference's own defect
        // is -1.02e-14; the guard catches a mistyped digit of y3.
        let mass = y_ref[0] + y_ref[1] + y_ref[2] - 1.0;
        assert!(
            mass.abs() <= 2.0e-14,
            "{ROBER}: y1 + y2 + y3 - 1 is {mass:e}, beyond 2e-14"
        );
        println!("  [answer-key] rober sum - 1 = {mass:e}");

        // HIRES, Table II.1.1 at t = 321.8122.
        assert_eq!(hires.rows.len(), 8, "{HIRES}: eight components");
        let hcol = |name: &str| {
            hires.column(name)
                .unwrap_or_else(|e| panic!("{HIRES}: the column {name}: {e}"))
        };
        let component = hcol("component");
        let t_end = hcol("t_end");
        let y0 = hcol("y0");
        let y_ref = hcol("y_ref");
        assert_eq!(
            component,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "{HIRES}: the component column"
        );
        for (i, t) in t_end.iter().enumerate() {
            assert_eq!(*t, 321.8122, "{HIRES}: row {i}: t_end");
        }
        assert_eq!(
            y0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
            "{HIRES}: the y0 column"
        );
        assert_eq!(y_ref[0], 7.371312573325668e-4, "{HIRES}: row 0: y_ref");
        assert_eq!(y_ref[7], 2.850001604814231e-3, "{HIRES}: row 7: y_ref");
        // HIRES conserves y7 + y8, the enzyme E free and bound (f7 = -f8 in
        // the report's section 1.2); the reference holds it to 0.0 in f64.
        let enzyme = y_ref[6] + y_ref[7] - 0.0057;
        assert!(
            enzyme.abs() <= 1.0e-17,
            "{HIRES}: y7 + y8 - 0.0057 is {enzyme:e}, beyond 1e-17"
        );
        println!("  [answer-key] hires y7 + y8 - 0.0057 = {enzyme:e}");
    }

    // ------------------------------------------- the TNF Flame D answer keys --

    /// The four Flame D keys, centreline first (SPEC-LIT section 137.5).
    const TNF_IDS: [&str; 4] = [
        "tnf-flameD-centreline",
        "tnf-flameD-x15",
        "tnf-flameD-x30",
        "tnf-flameD-x45",
    ];

    /// The 24 columns after the station column, in the archive's order, that
    /// all four keys share.
    const TNF_COLUMNS: [&str; 24] = [
        "f",
        "f_rms",
        "t",
        "t_rms",
        "y_o2",
        "y_o2_rms",
        "y_n2",
        "y_n2_rms",
        "y_h2",
        "y_h2_rms",
        "y_h2o",
        "y_h2o_rms",
        "y_ch4",
        "y_ch4_rms",
        "y_co_raman",
        "y_co_raman_rms",
        "y_co2",
        "y_co2_rms",
        "y_oh",
        "y_oh_rms",
        "y_no",
        "y_no_rms",
        "y_co_lif",
        "y_co_lif_rms",
    ];

    /// The atomic weights of SPEC-LIT section 137.5, g/mol. The O weight is
    /// that section's DESIGN choice, because the documentation gives none.
    const TNF_W_H: f64 = 1.008;
    const TNF_W_C: f64 = 12.011;
    const TNF_W_O: f64 = 15.999;
    /// The jet stream (1) and the coflow (2) element mass fractions that
    /// close (137.4), from SPEC-LIT section 137.5.
    const TNF_Y_H1: f64 = 0.0393;
    const TNF_Y_C1: f64 = 0.1170;
    const TNF_Y_H2: f64 = 0.0007;
    const TNF_Y_C2: f64 = 0.0;
    /// The stoichiometric mixture fraction of SPEC-LIT section 137.5.
    const TNF_F_STOIC: f64 = 0.351;

    /// All four Flame D keys, loaded and verified, in [`TNF_IDS`]'s order.
    fn tnf_keys() -> Vec<KeyFile> {
        TNF_IDS
            .iter()
            .map(|id| load(id).unwrap_or_else(|e| panic!("{id}: the key loads: {e}")))
            .collect()
    }

    /// One cell of a Flame D key, by data-row index and column name.
    fn tnf_cell(key: &KeyFile, row: usize, name: &str) -> f64 {
        let at = key
            .columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("{}: no column named {name}", key.id));
        key.rows[row][at]
    }

    /// The denominator of (137.4): the jet stream's `0.5 (Y_H1 - Y_H2)/W_H +
    /// 2 (Y_C1 - Y_C2)/W_C`, SPEC-LIT section 137.5.
    fn tnf_denominator() -> f64 {
        0.5 * (TNF_Y_H1 - TNF_Y_H2) / TNF_W_H + 2.0 * (TNF_Y_C1 - TNF_Y_C2) / TNF_W_C
    }

    /// The mixture fraction of one row by (137.4)-(137.5), SPEC-LIT section
    /// 137.5, from the row's Favre-mean species. `co` names the CO column,
    /// `"y_co_lif"` or `"y_co_raman"`. The H carriers are H2, H2O, CH4 and
    /// OH, the C carriers CH4, CO and CO2.
    fn tnf_mixture_fraction(key: &KeyFile, row: usize, co: &str) -> f64 {
        let y = |name: &str| tnf_cell(key, row, name);
        let w_h2 = 2.0 * TNF_W_H;
        let w_h2o = 2.0 * TNF_W_H + TNF_W_O;
        let w_ch4 = TNF_W_C + 4.0 * TNF_W_H;
        let w_oh = TNF_W_O + TNF_W_H;
        let w_co = TNF_W_C + TNF_W_O;
        let w_co2 = TNF_W_C + 2.0 * TNF_W_O;
        // (137.5): Y_H = sum_k n_H,k W_H Y_k / W_k, H2's term being y_h2 itself.
        let y_h = 2.0 * TNF_W_H * y("y_h2") / w_h2
            + 2.0 * TNF_W_H * y("y_h2o") / w_h2o
            + 4.0 * TNF_W_H * y("y_ch4") / w_ch4
            + TNF_W_H * y("y_oh") / w_oh;
        // (137.5): Y_C = sum_k n_C,k W_C Y_k / W_k.
        let y_c = TNF_W_C * y("y_ch4") / w_ch4
            + TNF_W_C * y(co) / w_co
            + TNF_W_C * y("y_co2") / w_co2;
        // (137.4): Bilger's F with the H and C element mass fractions only.
        (0.5 * (y_h - TNF_Y_H2) / TNF_W_H + 2.0 * (y_c - TNF_Y_C2) / TNF_W_C) / tnf_denominator()
    }

    /// The nine measured mass fractions of one row, summed: O2, N2, H2, H2O,
    /// CH4, CO (`co` names the column), CO2, OH and NO.
    fn tnf_mass_sum(key: &KeyFile, row: usize, co: &str) -> f64 {
        let y = |name: &str| tnf_cell(key, row, name);
        y("y_o2")
            + y("y_n2")
            + y("y_h2")
            + y("y_h2o")
            + y("y_ch4")
            + y(co)
            + y("y_co2")
            + y("y_oh")
            + y("y_no")
    }

    #[test]
    fn tnf_flame_d_keys_hold_their_columns_and_stations() {
        // answer-key: tnf-flameD-centreline
        // answer-key: tnf-flameD-x15
        // answer-key: tnf-flameD-x30
        // answer-key: tnf-flameD-x45
        let keys = tnf_keys();
        let [cl, x15, x30, x45] = &keys[..] else {
            panic!("four Flame D keys, found {}", keys.len());
        };

        // 1. The headers: the station column, then the 24 shared names.
        let header = |first: &str| -> Vec<String> {
            std::iter::once(first)
                .chain(TNF_COLUMNS)
                .map(str::to_string)
                .collect()
        };
        assert_eq!(cl.columns.len(), 25, "{}: 25 header names", cl.id);
        assert_eq!(cl.columns, header("x_d"), "{}: the header row", cl.id);
        for key in [x15, x30, x45] {
            assert_eq!(key.columns.len(), 25, "{}: 25 header names", key.id);
            assert_eq!(key.columns, header("r_d"), "{}: the header row", key.id);
        }

        // 2. The row counts.
        for (key, want) in [(cl, 16), (x15, 15), (x30, 15), (x45, 14)] {
            assert_eq!(key.rows.len(), want, "{}: the row count", key.id);
        }

        // 3. The centreline stations, x/d = 5 to 80 in steps of 5.
        let x_d: Vec<f64> = (1..=16).map(|i| 5.0 * f64::from(i)).collect();
        let got = cl.column("x_d").expect("the x_d column");
        assert_eq!(got, x_d, "{}: the x_d column", cl.id);

        // 4. The radial stations; x/d = 15 prints r/d = 0.28 twice.
        let r_x15 = [
            -0.56, -0.28, 0.0, 0.28, 0.28, 0.56, 0.83, 1.11, 1.39, 1.67, 1.94, 2.22, 2.5, 2.78,
            3.06,
        ];
        let r_x30 = [
            -0.83, -0.42, 0.0, 0.42, 0.83, 1.25, 1.67, 2.08, 2.5, 2.92, 3.33, 3.75, 4.17, 5.0,
            5.83,
        ];
        let r_x45 = [
            -1.11, -0.56, 0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.56, 6.67, 7.78,
        ];
        let r_d = |key: &KeyFile| key.column("r_d").expect("the r_d column");
        assert_eq!(r_d(x15), r_x15, "{}: the r_d column", x15.id);
        assert_eq!(r_d(x30), r_x30, "{}: the r_d column", x30.id);
        assert_eq!(r_d(x45), r_x45, "{}: the r_d column", x45.id);

        // 5. The pinned cells, the ones a retyping would bend first.
        let pin = |key: &KeyFile, row: usize, name: &str, want: f64| {
            assert_eq!(
                tnf_cell(key, row, name),
                want,
                "{}: row {row}: the {name} cell",
                key.id
            );
        };
        pin(cl, 0, "f", 0.9853);
        pin(cl, 0, "t", 298.0);
        pin(cl, 8, "x_d", 45.0);
        pin(cl, 8, "t", 1945.0);
        pin(cl, 8, "f", 0.3904);
        pin(cl, 15, "f", 0.1176);
        pin(cl, 15, "y_co_lif_rms", 2.45e-4);
        pin(x15, 3, "f", 0.9011);
        pin(x15, 4, "f", 0.9018);
        pin(x30, 2, "f", 0.6817);
        pin(x30, 2, "t", 1262.0);
        pin(x30, 14, "f", 0.0);
        pin(x45, 0, "t", 1852.0);
        pin(x45, 13, "y_co_lif_rms", 4.32e-5);

        // 6. Signs and ranges, over every row of the four keys.
        for key in &keys {
            for (row, cells) in key.rows.iter().enumerate() {
                let station = cells[0];
                for (name, &v) in key.columns.iter().zip(cells) {
                    let at = format!("{}: row {row} (station {station}): {name} = {v}", key.id);
                    match name.as_str() {
                        "f" => assert!((0.0..=1.0).contains(&v), "{at}: f lies in [0, 1]"),
                        "t" => assert!(
                            (290.0..=2100.0).contains(&v),
                            "{at}: t lies in [290, 2100] K"
                        ),
                        _ => {}
                    }
                    if name.ends_with("_rms") {
                        assert!(v >= 0.0, "{at}: an rms is not negative");
                    } else if name.starts_with("y_") {
                        assert!(v >= 0.0, "{at}: a mass fraction is not negative");
                    }
                }
            }
        }

        // 7. One line per key.
        for key in &keys {
            let first = key.rows.first().map_or(f64::NAN, |r| r[0]);
            let last = key.rows.last().map_or(f64::NAN, |r| r[0]);
            println!(
                "  [answer-key] tnf {}  {} rows  stations {first} to {last}",
                key.id,
                key.rows.len()
            );
        }
    }

    #[test]
    fn tnf_flame_d_rows_close_their_mass_and_mixture_fraction() {
        let keys = tnf_keys();
        // The oracle's denominator of (137.4), a guard on the typed constants
        // of SPEC-LIT section 137.5 before any row is read.
        let den = tnf_denominator();
        assert!(
            (den - 0.03862896676723586).abs() <= 1.0e-15,
            "the denominator of (137.4) is {den}, the oracle's is 0.03862896676723586"
        );
        for key in &keys {
            // (value, row) of the worst difference, for each CO column.
            let (mut lif_f, mut lif_s) = ((0.0_f64, 0_usize), (0.0_f64, 0_usize));
            let (mut raman_f, mut raman_s) = ((0.0_f64, 0_usize), (0.0_f64, 0_usize));
            for (row, cells) in key.rows.iter().enumerate() {
                let station = cells[0];
                let f = tnf_cell(key, row, "f");
                let df = (tnf_mixture_fraction(key, row, "y_co_lif") - f).abs();
                let ds = (tnf_mass_sum(key, row, "y_co_lif") - 1.0).abs();
                // The documentation says models are compared with the LIF CO,
                // and the Raman CO carries hydrocarbon interference. So the
                // Raman figures are printed and never asserted.
                let rf = (tnf_mixture_fraction(key, row, "y_co_raman") - f).abs();
                let rs = (tnf_mass_sum(key, row, "y_co_raman") - 1.0).abs();
                // 1. (137.4)-(137.5) rebuild the printed f (SPEC-LIT section 137.5).
                assert!(
                    df <= 5.0e-3,
                    "{}: row {row} (station {station}): (137.4)-(137.5) with CO from LIF give \
                     f = {}, printed {f}, differing by {df:e}, beyond 5e-3",
                    key.id,
                    tnf_mixture_fraction(key, row, "y_co_lif")
                );
                // 2. The nine measured mass fractions sum to one.
                assert!(
                    ds <= 2.0e-3,
                    "{}: row {row} (station {station}): the nine mass fractions with CO from \
                     LIF sum to {}, which is {ds:e} from 1, beyond 2e-3",
                    key.id,
                    tnf_mass_sum(key, row, "y_co_lif")
                );
                if df > lif_f.0 {
                    lif_f = (df, row);
                }
                if ds > lif_s.0 {
                    lif_s = (ds, row);
                }
                if rf > raman_f.0 {
                    raman_f = (rf, row);
                }
                if rs > raman_s.0 {
                    raman_s = (rs, row);
                }
            }
            println!(
                "  [answer-key] tnf mix {}  lif |F - f| {:.6e} (row {})  lif |sum - 1| {:.6e} \
                 (row {})  raman |F - f| {:.6e} (row {})  raman |sum - 1| {:.6e} (row {})",
                key.id, lif_f.0, lif_f.1, lif_s.0, lif_s.1, raman_f.0, raman_f.1, raman_s.0,
                raman_s.1
            );
        }
    }

    #[test]
    fn tnf_flame_d_centreline_anchors_the_chr16_band() {
        let keys = tnf_keys();
        let [cl, x15, x30, x45] = &keys[..] else {
            panic!("four Flame D keys, found {}", keys.len());
        };
        let col = |key: &KeyFile, name: &str| {
            key.column(name)
                .unwrap_or_else(|e| panic!("{}: the column {name}: {e}", key.id))
        };
        let x_d = col(cl, "x_d");
        let f = col(cl, "f");
        let t = col(cl, "t");

        // 1. T_max: the largest centreline temperature, once, at x/d = 45
        // (SPEC-LIT section 137.5's anchor for CHR-16's band).
        let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(t_max, 1945.0, "{}: the largest centreline t", cl.id);
        let at_max: Vec<usize> = t
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == t_max)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            at_max.len(),
            1,
            "{}: T_max occurs at rows {at_max:?}, not at exactly one row",
            cl.id
        );
        assert_eq!(
            x_d[at_max[0]], 45.0,
            "{}: row {}: T_max sits at x/d {}",
            cl.id, at_max[0], x_d[at_max[0]]
        );

        // 2. The stoichiometric length: f = 0.351 crossed between two
        // stations, interpolated linearly.
        let i = (0..f.len() - 1)
            .find(|&i| f[i] >= TNF_F_STOIC && TNF_F_STOIC > f[i + 1])
            .unwrap_or_else(|| panic!("{}: f never crosses {TNF_F_STOIC}", cl.id));
        assert_eq!(i, 8, "{}: f crosses {TNF_F_STOIC} after row {i}", cl.id);
        let l_stoic = x_d[i] + (x_d[i + 1] - x_d[i]) * (f[i] - TNF_F_STOIC) / (f[i] - f[i + 1]);
        assert!(
            (l_stoic - 47.5452196382429).abs() <= 1.0e-12,
            "{}: rows {i} and {}: L_stoic/d interpolates to {l_stoic}, the supervisor's f64 \
             value is 47.5452196382429",
            cl.id,
            i + 1
        );
        // 47.0 is the documentation's printed L_stoic/d for flame D; its
        // interpolant is not stated, and the measured difference is 0.545.
        assert!(
            (l_stoic - 47.0).abs() <= 1.0,
            "{}: rows {i} and {}: L_stoic/d is {l_stoic}, more than 1 from the printed 47.0",
            cl.id,
            i + 1
        );

        // 3. Each radial profile's r/d = 0 row repeats the centreline station
        // at its x/d, within the archive's own repeatability.
        let mut repeats = Vec::new();
        for (key, station) in [(x15, 15.0), (x30, 30.0), (x45, 45.0)] {
            let r_d = col(key, "r_d");
            let r0 = r_d
                .iter()
                .position(|&r| r == 0.0)
                .unwrap_or_else(|| panic!("{}: no row has r/d = 0", key.id));
            assert_eq!(r0, 2, "{}: the r/d = 0 row is row {r0}", key.id);
            let c = x_d
                .iter()
                .position(|&x| x == station)
                .unwrap_or_else(|| panic!("{}: no row has x/d = {station}", cl.id));
            let df = tnf_cell(key, r0, "f") - f[c];
            let dt = tnf_cell(key, r0, "t") - t[c];
            let rel = dt.abs() / t[c];
            assert!(
                df.abs() <= 0.015,
                "{}: row {r0} against {} row {c}: f differs by {df:+}, beyond 0.015",
                key.id,
                cl.id
            );
            assert!(
                rel <= 0.03,
                "{}: row {r0} against {} row {c}: t differs by {dt:+} K ({rel:e} relative), \
                 beyond 3 %",
                key.id,
                cl.id
            );
            repeats.push((key.id.clone(), df, dt, rel));
        }

        // 4. The coflow edge: the last station of each radial profile is air.
        for key in [x15, x30, x45] {
            let last = key.rows.len() - 1;
            let (f_edge, t_edge) = (tnf_cell(key, last, "f"), tnf_cell(key, last, "t"));
            assert!(
                f_edge <= 0.005,
                "{}: row {last}: f = {f_edge} at the coflow edge, beyond 0.005",
                key.id
            );
            assert!(
                t_edge <= 330.0,
                "{}: row {last}: t = {t_edge} K at the coflow edge, beyond 330 K",
                key.id
            );
        }

        // 5. The printed lines.
        println!(
            "  [answer-key] tnf T_max {t_max:.0} K at x/d {:.0}, L_stoic/d {l_stoic:.3} (printed 47.0)",
            x_d[at_max[0]]
        );
        for (id, df, dt, rel) in &repeats {
            println!("  [answer-key] tnf {id}  r/d = 0 against the centreline  df {df:+.4}  dT {dt:+} K  rel {rel:.4e}");
        }
    }
}
