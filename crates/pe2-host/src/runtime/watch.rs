//! Debugging aids on PSX memory.
//!
//! `PE2_WATCH=0x801D49C4[,addr...]` (up to four) arms the x86 debug
//! registers on the game and IRQ threads; every write to a watched word logs
//! the writing instruction's address and the new value. Map the address to a
//! function with tools/symbolize.py.
//!
//! `PE2_PEEK=0x80082818:0x58[,addr:len...]` prints those bytes of PSX RAM
//! (or of the scratchpad) every five seconds (a game's state while it seems stuck).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use windows_sys::Win32::Foundation::{EXCEPTION_SINGLE_STEP, HANDLE, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, CONTEXT_DEBUG_REGISTERS_X86, EXCEPTION_POINTERS, GetThreadContext,
    SetThreadContext,
};
use windows_sys::Win32::System::Threading::{ResumeThread, SuspendThread};

static WATCHED: OnceLock<Vec<u32>> = OnceLock::new();

/// Widescreen: what the game says of its frames (`pcMenuFrame` in the fork's
/// menu.c reports it every frame: 1 live field gameplay, 2 the field held for
/// a moment, 3 PE2PC's menu open, 0 other frames: a camera cut's, a load's),
/// when it last said anything, and when it last said 1 (ms since first use,
/// plus 1, so 0 means never).
static FIELD: AtomicU32 = AtomicU32::new(0);
static FIELD_AT: AtomicU64 = AtomicU64::new(0);
static LIVE_AT: AtomicU64 = AtomicU64::new(0);

/// How long after live field gameplay other frames still count as the
/// field: the field held (state 2, a room's transition), and other frames
/// (state 0: a camera cut's, 124-238 ms measured; the end of the attract
/// demo, about 520 ms, is not).
const FIELD_HOLD_MS: u64 = 600;
const OTHER_HOLD_MS: u64 = 260;

fn field_clock() -> u64 {
    static BASE: OnceLock<Instant> = OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// Called by the game each frame it runs `pcMenuFrame`.
#[unsafe(no_mangle)]
pub extern "C" fn pc_widescreen_field(field: i32) {
    // Not interrupted while printing.
    let _g = super::irq::guard();
    let state = field.clamp(0, 3) as u32;
    if FIELD.swap(state, Ordering::Relaxed) != state {
        let name = ["off", "ON", "held", "menu"][state as usize];
        eprintln!("[wide {:8.3}] field gameplay -> {name}", super::spu::stamp());
    }
    let now = field_clock();
    FIELD_AT.store(now, Ordering::Relaxed);
    if state == 1 {
        LIVE_AT.store(now, Ordering::Relaxed);
    }
}

/// Live field gameplay, as of the last 150 ms.
fn field_live() -> bool {
    let at = FIELD_AT.load(Ordering::Relaxed);
    FIELD.load(Ordering::Relaxed) == 1 && at != 0 && field_clock().saturating_sub(at) < 150
}

/// Field gameplay, and no movie streaming or menu open: live (as of the
/// last 150 ms), or live no longer than FIELD_HOLD_MS ago.
fn in_field() -> bool {
    if super::stream::active() {
        return false;
    }
    let now = field_clock();
    let at = FIELD_AT.load(Ordering::Relaxed);
    match FIELD.load(Ordering::Relaxed) {
        1 if at != 0 && now.saturating_sub(at) < 150 => true,
        3 => false,
        state => {
            let live = LIVE_AT.load(Ordering::Relaxed);
            let hold = if state == 2 { FIELD_HOLD_MS } else { OTHER_HOLD_MS };
            live != 0 && now.saturating_sub(live) < hold
        }
    }
}

/// gDisplayState (screenDistance, at +0x110, is 0x80071078).
const DISPLAY_STATE: u32 = 0x8007_0F68;

/// One of the game's own menus has the screen: the display handed to a
/// transition task (displayOwner 1) for a game menu (pendingMode 0x40-0x4F,
/// measured 0x41 for the item menu; not the map, 0x43) or a descriptor task
/// (0x81: the save prompt).
fn game_menu() -> bool {
    // SAFETY: fixed addresses inside PSX RAM, mapped for the process's life.
    let rd8 = |o: u32| unsafe { ((DISPLAY_STATE + o) as usize as *const u8).read() };
    let (owner, mode) = (rd8(0x1E), rd8(0x10D));
    owner == 1 && ((mode & 0xF0 == 0x40 && mode != 0x43) || mode == 0x81)
}

/// A game menu opened from field play (see `game_menu`), until it closes:
/// the frozen field stays widescreen behind it.
pub fn menu_over_field() -> bool {
    static OVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !game_menu() {
        // Closed: the field it returns to counts as live from now (its
        // first frame is held, long after the last live one).
        if OVER.swap(false, Ordering::Relaxed) {
            LIVE_AT.store(field_clock(), Ordering::Relaxed);
        }
        return false;
    }
    if !OVER.load(Ordering::Relaxed) && in_field() {
        OVER.store(true, Ordering::Relaxed);
        eprintln!("[wide {:8.3}] game menu over the field", super::spu::stamp());
    }
    OVER.load(Ordering::Relaxed)
}

fn watched() -> &'static [u32] {
    WATCHED.get_or_init(|| {
        std::env::var("PE2_WATCH")
            .map(|v| {
                v.split(',')
                    .filter_map(|a| u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                    .take(4)
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub fn enabled() -> bool {
    !watched().is_empty()
}

/// Arms the watchpoints on `thread` (not the calling thread).
pub fn arm(thread: HANDLE) {
    let addrs = watched();
    if addrs.is_empty() {
        return;
    }
    // SAFETY: suspends another thread, edits its debug registers, resumes.
    unsafe {
        SuspendThread(thread);
        let mut ctx: CONTEXT = std::mem::zeroed();
        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS_X86;
        GetThreadContext(thread, &mut ctx);
        let mut dr7 = 0u32;
        for (i, &a) in addrs.iter().enumerate() {
            match i {
                0 => ctx.Dr0 = a & !3,
                1 => ctx.Dr1 = a & !3,
                2 => ctx.Dr2 = a & !3,
                _ => ctx.Dr3 = a & !3,
            }
            // Local enable, break on write (01), length 4 bytes (11).
            dr7 |= 1 << (i * 2);
            dr7 |= 0b1101 << (16 + i * 4);
        }
        ctx.Dr7 = dr7;
        SetThreadContext(thread, &ctx);
        ResumeThread(thread);
    }
}

/// Widescreen pan probe. `PE2_AYA_PROBE=1` logs, twice a second, the
/// translation of the matrix `gPlayerStatus.coordMtx` points at, the one
/// 0x20 bytes after it (the root node's `workm`, if `coordMtx` points at its
/// `coord`), the projection distance, and where `workm` would project.
fn start_aya_probe() {
    if !std::env::var("PE2_AYA_PROBE").is_ok_and(|v| v.trim() == "1") {
        return;
    }
    std::thread::spawn(|| {
        eprintln!("[aya] probe started");
        const COORD_MTX: u32 = 0x8007_3B8C; // gPlayerStatus + 4
        const SCREEN_DISTANCE: u32 = 0x8007_1078; // gDisplayState + 0x110
        // SAFETY: fixed addresses inside PSX RAM, mapped for the process's
        // life; a racy read is fine for a look.
        let rd32 = |a: u32| unsafe { (a as usize as *const i32).read_unaligned() };
        let rd16 = |a: u32| unsafe { (a as usize as *const u16).read_unaligned() };
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let ptr = rd32(COORD_MTX) as u32;
            let h = rd16(SCREEN_DISTANCE);
            if !(0x8000_0000..0x8020_0000 - 0x40).contains(&ptr) {
                eprintln!("[aya {:8.3}] coordMtx=0x{ptr:08X} (not in RAM) H={h}", super::spu::stamp());
                continue;
            }
            let t = |base: u32| [rd32(base + 0x14), rd32(base + 0x18), rd32(base + 0x1C)];
            let local = t(ptr);
            let work = t(ptr + 0x20);
            let proj = if work[2] > 0 {
                let z = work[2] as i64;
                format!("({}, {})", work[0] as i64 * h as i64 / z, work[1] as i64 * h as i64 / z)
            } else {
                "behind".to_string()
            };
            eprintln!(
                "[aya {:8.3}] coordMtx=0x{ptr:08X} local t=({}, {}, {}) work t=({}, {}, {}) H={h} proj={proj}",
                super::spu::stamp(),
                local[0],
                local[1],
                local[2],
                work[0],
                work[1],
                work[2]
            );
        }
    });
}

/// Widescreen pan: where Aya's root is on the screen, as a fraction of the
/// picture's height. Err: not the field, or the field live with no player in
/// it (the title screen); Ok(None): the field without a reading now (held
/// where it was, as across a camera cut). Read on the game's thread when a
/// frame is complete (gpu's ClearOTagR and DrawOTag), so the frame shown
/// carries its own (`frame_row`).
pub type Row = Result<Option<f32>, ()>;

pub fn aya_row() -> Row {
    const COORD_MTX: u32 = 0x8007_3B8C; // gPlayerStatus + 4
    const SCREEN_DISTANCE: u32 = 0x8007_1078; // gDisplayState + 0x110
    // A game menu over the frozen field: held where it was (infinity).
    if menu_over_field() {
        return Ok(Some(f32::INFINITY));
    }
    if !in_field() {
        return Err(());
    }
    // SAFETY: fixed addresses inside PSX RAM, mapped for the process's life.
    let rd32 = |a: u32| unsafe { (a as usize as *const i32).read_unaligned() };
    let rd16 = |a: u32| unsafe { (a as usize as *const u16).read_unaligned() };
    let none = || if field_live() { Err(()) } else { Ok(None) };
    let ptr = rd32(COORD_MTX) as u32;
    let h = rd16(SCREEN_DISTANCE) as i64;
    if !(0x8000_0000..0x8020_0000 - 0x40).contains(&ptr) || h == 0 {
        return none();
    }
    // The root's `workm` is 0x20 bytes after its `coord`; its translation
    // (view space) is at +0x14: x, y, z.
    let y = rd32(ptr + 0x20 + 0x18) as i64;
    let z = rd32(ptr + 0x20 + 0x1C) as i64;
    if z <= 0 {
        return none();
    }
    let row = 120.0 + (y * h / z) as f32;
    let extent = aya_extent(ptr, h);
    PENDING_EXTENT.0.store(extent.map_or(f32::NAN, |e| e.0).to_bits(), Ordering::Relaxed);
    PENDING_EXTENT.1.store(extent.map_or(f32::NAN, |e| e.1).to_bits(), Ordering::Relaxed);
    Ok(Some((row / 240.0).clamp(-1.0, 2.0)))
}

/// Aya's extent read with the last `aya_row` (top and bottom, fractions of
/// the picture's height; NaN: none), published with its row by `frame_row`.
static PENDING_EXTENT: (AtomicU32, AtomicU32) = (AtomicU32::new(0x7FC0_0000), AtomicU32::new(0x7FC0_0000));

/// Where Aya's body reaches on the screen, from the top of her head to her
/// feet, as fractions of the picture's height. Her model's parts
/// (`GfxCoord`, 0x50 bytes each) follow its `TmdObject` (0x34 bytes: its
/// `coords` at +8, `partCount` at +0x30), the root first, whose `coord`
/// `coordMtx` points at (+4); each part's `workm` (+0x24) holds where it is
/// in view space, its translation at +0x14. The parts are joints: the top
/// of her head is above the highest (her head's), her soles a little below
/// the lowest. None when the layout does not check out.
fn aya_extent(coord_mtx: u32, h: i64) -> Option<(f32, f32)> {
    const PART_BYTES: u32 = 0x50;
    const OBJECT_BYTES: u32 = 0x34;
    // Beyond the joints, as fractions of the span between them: the top of
    // the head over the head's joint, the soles under the ankles'.
    const ABOVE: f32 = 0.14;
    const BELOW: f32 = 0.06;
    // SAFETY: addresses checked to lie inside PSX RAM, mapped for the
    // process's life; a racy read is fine for a reading.
    let rd32 = |a: u32| unsafe { (a as usize as *const i32).read_unaligned() };
    let in_ram = |a: u32, len: u32| (0x8000_0000..0x8020_0000u32.saturating_sub(len)).contains(&a);
    let parts = coord_mtx.wrapping_sub(4);
    let object = parts.wrapping_sub(OBJECT_BYTES);
    if !in_ram(object, OBJECT_BYTES) || rd32(object + 8) as u32 != parts {
        return None;
    }
    let count = rd32(object + 0x30);
    if !(1..=64).contains(&count) || !in_ram(parts, count as u32 * PART_BYTES) {
        return None;
    }
    let (mut top, mut bottom) = (f32::INFINITY, f32::NEG_INFINITY);
    // PE2_TRACE=ayaparts: every part's place, once a second.
    static LOGGED: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    let log = super::trace("ayaparts") && {
        let mut at = LOGGED.lock().unwrap();
        let due = at.is_none_or(|t| t.elapsed().as_secs_f32() >= 1.0);
        if due {
            *at = Some(Instant::now());
        }
        due
    };
    let mut line = String::new();
    for i in 0..count as u32 {
        let node = parts + i * PART_BYTES;
        let t = node + 0x24 + 0x14;
        let (y, z) = (rd32(t + 4) as i64, rd32(t + 8) as i64);
        if log {
            let parent = rd32(node + 0x4C) as u32;
            let p = if parent >= parts && parent < parts + count as u32 * PART_BYTES {
                format!("{}", (parent - parts) / PART_BYTES)
            } else {
                format!("{parent:08X}")
            };
            line += &format!(
                " #{i}<{p} stamp {:08X} y {y} z {z} row {:.3};",
                rd32(node) as u32,
                if z > 0 { (120.0 + (y * h / z) as f32) / 240.0 } else { f32::NAN }
            );
        }
        if z <= 0 {
            continue;
        }
        let row = (120.0 + (y * h / z) as f32) / 240.0;
        top = top.min(row);
        bottom = bottom.max(row);
    }
    if log {
        eprintln!(
            "[ayaparts {:8.3}] H {h}, strip at {:.3}, {count} parts:{line}",
            super::spu::stamp(),
            f32::from_bits(crate::present::STRIP_TOP.load(Ordering::Relaxed))
        );
    }
    // No part in front of the camera: no extent.
    if !top.is_finite() {
        return None;
    }
    let span = bottom - top;
    Some((top - span * ABOVE, bottom + span * BELOW))
}

/// The pan tracker's state (see `frame_row`).
struct Track {
    /// The last reading, when it was taken, and the background shown then.
    good: Option<f32>,
    at: Option<Instant>,
    good_background: Option<u64>,
    /// The shown frame's background, frames shown since it changed, and
    /// jumps ignored in a row.
    background: Option<u64>,
    since_cut: u32,
    ignored: u32,
}
static TRACK: std::sync::Mutex<Track> = std::sync::Mutex::new(Track {
    good: None,
    at: None,
    good_background: None,
    background: None,
    since_cut: u32::MAX,
    ignored: 0,
});

/// PE2_TRACE=panshots: whether the frame being shown is near a camera cut
/// (held out of live play, or within ten frames after a cut).
pub fn near_cut() -> bool {
    (FIELD.load(Ordering::Relaxed) != 1 && in_field()) || TRACK.lock().unwrap().since_cut < 10
}

/// A new frame is shown, with Aya's place now (see `aya_row`) and the
/// fingerprint of its background: publishes Aya's place to `present` (NaN
/// when there is no player). A camera cut is the background changing: a jump
/// of Aya's place (more than 15% of the height) is taken on that frame or
/// the few after it, when the game's state catches up, and the strip goes
/// there at once. A jump with the same background is the game's state
/// ahead of the picture (it shows the old view for a moment at a cut):
/// ignored, unless it lasts. A place far off the picture is no reading.
///
/// Not the field (Err) while the picture is still the one Aya was last read
/// on (the same background, within FIELD_HOLD_MS) is held: entering a door,
/// the game stalls on the frozen room (measured 370 ms, its first frame
/// after without a player) and hands the screen to its fade (displayOwner
/// 2, 340 ms after the last live frame); each was a 4:3 frame. Returns
/// whether the field is off (the crop off: splits from before are stale).
pub fn frame_row(row: Row, background: Option<u64>) -> bool {
    use std::sync::atomic::Ordering::{Relaxed, Release};
    const JUMP: f32 = 0.15;
    const CATCH_UP_FRAMES: u32 = 4;
    const LASTING_FRAMES: u32 = 8;
    let publish = |v: f32| crate::present::AYA_FY.store(v.to_bits(), Relaxed);
    let pan = super::trace("pan");
    let mut t = TRACK.lock().unwrap();
    t.since_cut = t.since_cut.saturating_add(1);
    if background.is_some() && background != t.background {
        if t.background.is_some() && pan {
            eprintln!("[pan {:8.3}] new background shown", super::spu::stamp());
        }
        (t.background, t.since_cut) = (background, 0);
    }
    if pan {
        let state = ["off", "ON", "held", "menu"][FIELD.load(Relaxed).min(3) as usize];
        let row = match row {
            Err(()) => "none".to_string(),
            Ok(None) => "unread".to_string(),
            Ok(Some(v)) => format!(
                "{v:.3} (head {:.3}, feet {:.3})",
                f32::from_bits(PENDING_EXTENT.0.load(Relaxed)),
                f32::from_bits(PENDING_EXTENT.1.load(Relaxed))
            ),
        };
        // gDisplayState's displayOwner (+0x1E), pendingMode (+0x10D),
        // height (+0x1A) and gameMode (+0x11E).
        // SAFETY: fixed addresses inside PSX RAM, mapped for the process's life.
        let rd8 = |o: u32| unsafe { ((DISPLAY_STATE + o) as usize as *const u8).read() };
        let height = unsafe { ((DISPLAY_STATE + 0x1A) as usize as *const i16).read_unaligned() };
        eprintln!(
            "[pan {:8.3}] frame: field {state}, Aya at {row}, owner {} mode 0x{:02X} height {height} gameMode {}",
            super::spu::stamp(),
            rd8(0x1E),
            rd8(0x10D),
            rd8(0x11E)
        );
    }
    // A game menu over the field (aya_row): the strip held, nothing else
    // changed.
    if row == Ok(Some(f32::INFINITY)) {
        publish(f32::INFINITY);
        return false;
    }
    let frozen = t.good.is_some()
        && t.background == t.good_background
        && t.at.is_some_and(|at| (at.elapsed().as_millis() as u64) < FIELD_HOLD_MS);
    if row.is_err() && frozen {
        if pan {
            eprintln!("[pan {:8.3}] not the field, on the same picture: held", super::spu::stamp());
        }
        return false;
    }
    let on_picture = |v: &f32| (-0.25..=1.25).contains(v);
    let off_picture = matches!(row, Ok(Some(v)) if !on_picture(&v));
    let row = row.map(|r| r.filter(on_picture));
    match row {
        Err(()) => {
            (t.good, t.at, t.ignored) = (None, None, 0);
            publish(f32::NAN);
            return true;
        }
        Ok(None) => {
            if t.at.is_none_or(|at| at.elapsed().as_millis() as u64 >= FIELD_HOLD_MS) {
                (t.good, t.at, t.ignored) = (None, None, 0);
                // Aya read off the picture for a while (a scene's dialogue
                // looking elsewhere): the crop stays, the strip held where
                // it is (infinity). No reading at all: no player (NaN).
                publish(if off_picture { f32::INFINITY } else { f32::NAN });
            }
        }
        Ok(Some(v)) => {
            let jump = t.good.is_some_and(|g| (v - g).abs() > JUMP);
            if jump && t.since_cut >= CATCH_UP_FRAMES && t.ignored < LASTING_FRAMES {
                t.ignored += 1;
                if pan {
                    eprintln!("[pan {:8.3}] jump with the same background: ignored", super::spu::stamp());
                }
                return false;
            }
            let cut = jump || t.good.is_none();
            (t.good, t.at, t.ignored) = (Some(v), Some(Instant::now()), 0);
            t.good_background = t.background;
            // Her extent, read with this row, published before it.
            crate::present::AYA_TOP.store(PENDING_EXTENT.0.load(Relaxed), Relaxed);
            crate::present::AYA_BOTTOM.store(PENDING_EXTENT.1.load(Relaxed), Relaxed);
            publish(v);
            if cut {
                crate::present::AYA_CUTS.fetch_add(1, Release);
                if pan {
                    eprintln!("[pan {:8.3}] cut", super::spu::stamp());
                }
            }
        }
    }
    false
}

/// Starts PE2_PEEK, if set.
pub fn start_peek() {
    start_aya_probe();
    let Ok(spec) = std::env::var("PE2_PEEK") else { return };
    let regions: Vec<(u32, u32)> = spec
        .split(',')
        .filter_map(|r| {
            let (a, l) = r.split_once(':')?;
            let num = |v: &str| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok();
            Some((num(a)?, num(l)?.min(0x400)))
        })
        .filter(|&(a, l)| {
            let inside = |lo: u32, hi: u32| (lo..hi).contains(&a) && a + l <= hi;
            inside(0x8000_0000, 0x8020_0000) || inside(0x1F80_0000, 0x1F80_0400)
        })
        .collect();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            for &(addr, len) in &regions {
                // SAFETY: inside PSX RAM, mapped for the process's life; a
                // racy read of bytes is fine for a look.
                let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) };
                let hex: Vec<String> =
                    bytes.chunks(4).map(|w| w.iter().map(|b| format!("{b:02X}")).collect()).collect();
                eprintln!("[peek {:8.3}] {addr:08X}: {}", super::spu::stamp(), hex.join(" "));
            }
        }
    });
}

pub fn install_handler() {
    if enabled() {
        // SAFETY: registering a handler with the documented signature.
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
    }
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the OS passes valid exception pointers.
    let (rec, ctx) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    if rec.ExceptionCode != EXCEPTION_SINGLE_STEP as NTSTATUS {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    let hit = ctx.Dr6 & 0xF;
    for (i, &a) in watched().iter().enumerate() {
        if hit & (1 << i) != 0 {
            // SAFETY: the watched word is in mapped PSX RAM.
            let value = unsafe { ((a & !3) as *const u32).read_volatile() };
            let mips = super::mips::CUR_PC.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[watch] write to 0x{a:08X} = 0x{value:08X} by eip 0x{:08X} (last MIPS pc 0x{mips:08X})",
                ctx.Eip
            );
        }
    }
    ctx.Dr6 = 0;
    -1 // EXCEPTION_CONTINUE_EXECUTION
}
