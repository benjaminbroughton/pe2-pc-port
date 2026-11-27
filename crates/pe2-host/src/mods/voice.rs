//! Voiced dialogue (prototype): a mod's recording plays with a dialogue line.
//!
//! A line is known as the translations know it: its dialogue file's hash and
//! its record (the record's offset in that file, as `dumps/text/` gives
//! `record = 0x117C`). A mod supplies `voice/<cap hash>/<record>.ogg` (or
//! .flac, .wav; the record in hexadecimal, with or without `0x`):
//!
//! ```text
//! mods/my-voices/voice/1f9d66fcd224f2dd/117C.ogg
//! ```
//!
//! The game's dialogue playback (captions.c in the fork) tells us when a
//! line starts (`pc_cap_voice_line`) and when it is left
//! (`pc_cap_voice_stop`): a line's clip starts with its text, and stops
//! when the player moves on or the dialogue ends or is skipped. A timed
//! line (one that moves on by itself) is held on screen until its clip has
//! finished (`pc_cap_voice_playing`). Clips are mixed like the music's
//! recordings, without reverb.
//!
//! `PE2_TRACE=voice` names every line as it starts, with or without a clip,
//! which is how to find the file and record a recording is for.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// The clips mods supply: (cap hash, record offset) -> file.
fn index() -> &'static HashMap<(u64, usize), PathBuf> {
    static INDEX: OnceLock<HashMap<(u64, usize), PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut out = HashMap::new();
        if !super::enabled() {
            return out;
        }
        // Lowest priority first, so the highest overwrites it.
        for m in super::installed().into_iter().filter(|m| m.enabled).rev() {
            let Ok(caps) = std::fs::read_dir(m.dir.join("voice")) else { continue };
            for cap in caps.flatten().filter(|e| e.path().is_dir()) {
                let Some(hash) = cap.file_name().to_str().and_then(|n| u64::from_str_radix(n, 16).ok()) else {
                    continue;
                };
                let Ok(clips) = std::fs::read_dir(cap.path()) else { continue };
                for clip in clips.flatten() {
                    let path = clip.path();
                    let ext = path.extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase);
                    if !matches!(ext.as_deref(), Some("ogg" | "flac" | "wav")) {
                        continue;
                    }
                    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
                    let digits = stem.trim_start_matches("0x").trim_start_matches("0X");
                    let Ok(record) = usize::from_str_radix(digits, 16) else { continue };
                    out.insert((hash, record), path);
                }
            }
        }
        if !out.is_empty() {
            eprintln!("pe2: mods: {} voice clips", out.len());
        }
        out
    })
}

/// Dialogue files the game has loaded, by where they are (the file, or the
/// translated copy the game was given instead): their hash.
fn loaded() -> &'static Mutex<HashMap<usize, u64>> {
    static LOADED: OnceLock<Mutex<HashMap<usize, u64>>> = OnceLock::new();
    LOADED.get_or_init(Default::default)
}

/// Whether anything here can play (or is traced): else the hooks do nothing.
fn active() -> bool {
    !index().is_empty() || crate::runtime::trace("voice")
}

/// Whether translate.rs should tell us the hash of each dialogue file loaded.
pub fn wanted() -> bool {
    active()
}

/// From translate.rs, as the game loads dialogue file `file`, of `hash`.
pub fn note_cap(file: *const u8, hash: u64) {
    if active() {
        loaded().lock().unwrap().insert(file as usize, hash);
    }
}

/// From translate.rs: the game uses `copy` in place of `file`.
pub fn note_copy(file: *const u8, copy: *const u8) {
    if active() {
        let mut loaded = loaded().lock().unwrap();
        if let Some(&hash) = loaded.get(&(file as usize)) {
            loaded.insert(copy as usize, hash);
        }
    }
}

struct Playing {
    samples: Arc<Vec<(i16, i16)>>,
    position: usize,
}

#[derive(Default)]
struct Player {
    playing: Option<Playing>,
    /// Decoded clips (they are short; kept once played).
    ready: HashMap<(u64, usize), Arc<Vec<(i16, i16)>>>,
}

static PLAYER: Mutex<Option<Player>> = Mutex::new(None);

fn with_player<R>(f: impl FnOnce(&mut Player) -> R) -> R {
    f(PLAYER.lock().unwrap().get_or_insert_with(Player::default))
}

/// The game starts showing the line whose record is at `record` in the
/// dialogue file at `file`.
#[unsafe(no_mangle)]
pub extern "C" fn pc_cap_voice_line(file: *const u8, record: *const u8) {
    if !active() || file.is_null() || record.is_null() {
        return;
    }
    let _g = crate::runtime::irq::guard();
    let offset = (record as usize).wrapping_sub(file as usize);
    let hash = loaded().lock().unwrap().get(&(file as usize)).copied();
    let clip = hash.and_then(|h| index().get(&(h, offset)).map(|p| ((h, offset), p)));
    if crate::runtime::trace("voice") {
        let name = hash.map_or("an unknown file".to_string(), |h| format!("{h:016x}"));
        let has = clip.map_or("no clip".to_string(), |(_, p)| p.display().to_string());
        eprintln!("[voice] line {name} record 0x{offset:X}: {has}");
    }
    let Some((key, path)) = clip else {
        with_player(|p| p.playing = None);
        return;
    };
    let samples = match with_player(|p| p.ready.get(&key).cloned()) {
        Some(s) => s,
        // Short clips: decoded here, the first time each plays.
        None => match super::music::decode(path) {
            Ok((samples, _)) => {
                let s = Arc::new(samples);
                with_player(|p| p.ready.insert(key, s.clone()));
                s
            }
            Err(e) => {
                eprintln!("pe2: mods: {}: {e}", path.display());
                return;
            }
        },
    };
    with_player(|p| p.playing = Some(Playing { samples, position: 0 }));
}

/// The game leaves the line (the player moved on, or the dialogue ended).
#[unsafe(no_mangle)]
pub extern "C" fn pc_cap_voice_stop() {
    if !active() {
        return;
    }
    let _g = crate::runtime::irq::guard();
    with_player(|p| p.playing = None);
}

/// 1 while a line's clip is still playing (a timed line waits for it).
#[unsafe(no_mangle)]
pub extern "C" fn pc_cap_voice_playing() -> i32 {
    if index().is_empty() {
        return 0;
    }
    let _g = crate::runtime::irq::guard();
    with_player(|p| i32::from(p.playing.as_ref().is_some_and(|c| c.position < c.samples.len())))
}

/// Adds the playing clip to `out` (the SPU's next `frames` samples of mods'
/// sound).
pub fn mix(out: &mut VecDeque<(i16, i16)>, frames: usize) {
    let mut guard = PLAYER.lock().unwrap();
    let Some(c) = guard.as_mut().and_then(|p| p.playing.as_mut()) else { return };
    while out.len() < frames {
        out.push_back((0, 0));
    }
    for o in out.iter_mut().take(frames) {
        let (l, r) = c.samples.get(c.position).copied().unwrap_or((0, 0));
        *o = (o.0.saturating_add(l), o.1.saturating_add(r));
        c.position = (c.position + 1).min(c.samples.len());
    }
}
