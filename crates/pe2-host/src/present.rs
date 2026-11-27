//! Shows the game's picture in the window through the graphics card,
//! letterboxed at 4:3: the hardware layer's latched picture, already on the
//! card, or else the CPU's (sent as a texture). Optionally as a CRT
//! television would ([video] crt): scanlines and an aperture grille; and
//! at whole multiples of its height ([video] integer_scaling).

use std::sync::Arc;

use winit::window::Window;

use crate::gfx::Gfx;

const SHADER: &str = r#"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the viewport.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return o;
}

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct Params {
    // The console's lines in the picture (240, or 480 interlaced).
    lines: f32,
    // 0: the picture fills the viewport; 1: the same, with whole texels
    // (shown at whole multiples of its size); 2: widescreen, a strip of it
    // fills the viewport (`visible` of its height, from `top` down).
    mode: f32,
    // The picture's size in texels.
    size: vec2<f32>,
    visible: f32,
    top: f32,
    // Widescreen in layers: the strip is of `scene`, and `frame` (the
    // whole picture) shows over it where `ui` marks the UI, in a 4:3
    // rectangle in the middle of the viewport.
    layered: f32,
    _pad: f32,
};
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var scene: texture_2d<f32>;
@group(0) @binding(4) var ui: texture_2d<f32>;

// The hardware layer keeps channels in the console's scale (5 bits times
// 8, up to 248): `gain` brings them to the full range.
override gain: f32 = 1.0;
// A CRT television's look.
override crt: bool = false;

// How much of a scanline's light reaches `d` lines from its centre: a
// gaussian beam, wider where the line is brighter, as a tube's spot is.
fn beam(d: f32, c: vec3<f32>) -> f32 {
    let luma = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let sigma = mix(0.26, 0.42, luma);
    return exp(-(d * d) / (2.0 * sigma * sigma));
}

// The picture at `uv` of the viewport.
fn picture(uv: vec2<f32>) -> vec3<f32> {
    if (params.mode < 1.5) {
        return textureSampleLevel(frame, smp, uv, 0.0).rgb;
    }
    let strip = vec2<f32>(uv.x, uv.y * params.visible + params.top);
    if (params.layered < 0.5) {
        return textureSampleLevel(frame, smp, strip, 0.0).rgb;
    }
    var c = textureSampleLevel(scene, smp, strip, 0.0).rgb;
    // The 4:3 rectangle is `visible` of the viewport's width.
    let at = vec2<f32>((uv.x - (1.0 - params.visible) * 0.5) / params.visible, uv.y);
    if (at.x >= 0.0 && at.x <= 1.0) {
        let m = textureSampleLevel(ui, smp, at, 0.0).r;
        c = mix(c, textureSampleLevel(frame, smp, at, 0.0).rgb, m);
    }
    return c;
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    var c: vec3<f32>;
    if (crt) {
        // The two scanlines around this point, each sampled at its centre
        // and weighted by its beam (a strip has fewer of them).
        let n = params.lines * select(1.0, params.visible, params.mode > 1.5);
        let y = o.uv.y * n - 0.5;
        let i = floor(y);
        let f = y - i;
        let c0 = picture(vec2<f32>(o.uv.x, (i + 0.5) / n)) * gain;
        let c1 = picture(vec2<f32>(o.uv.x, (i + 1.5) / n)) * gain;
        c = (c0 * beam(f, c0) + c1 * beam(1.0 - f, c1)) * 1.45;
        // Aperture grille: alternate red, green and blue stripes, column by
        // column of the window.
        let col = u32(o.pos.x) % 3u;
        let mask = select(select(vec3<f32>(0.82, 0.82, 1.08), vec3<f32>(0.82, 1.08, 0.82), col == 1u),
                          vec3<f32>(1.08, 0.82, 0.82), col == 0u);
        c = c * mask;
    } else if (params.mode > 0.5 && params.mode < 1.5) {
        let size = params.size;
        c = textureLoad(frame, vec2<i32>(min(floor(o.uv * size), size - 1.0)), 0).rgb * gain;
    } else {
        c = picture(o.uv) * gain;
    }
    return vec4<f32>(min(c, vec3<f32>(1.0)), 1.0);
}
"#;

pub struct Presenter {
    gfx: &'static Gfx,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// For CPU pictures, and for the hardware layer's (with the gain); each
    /// without and with the CRT look (a setting that may change at any
    /// time).
    pipeline: [wgpu::RenderPipeline; 2],
    card_pipeline: [wgpu::RenderPipeline; 2],
    /// The shader's parameters (see `params` there), and what they hold.
    params: wgpu::Buffer,
    params_now: [f32; PARAMS],
    no_ui: wgpu::TextureView,
    /// The bind group of the hardware layer's picture, by its size.
    card: Option<((u32, u32), wgpu::BindGroup)>,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The picture's texture, its size and its bind group.
    frame: Option<(wgpu::Texture, (u32, u32), wgpu::BindGroup)>,
    pixels: Vec<u32>,
    /// The surface wants configuring again (it said it was suboptimal):
    /// done before the next frame, never while one of its frames is held.
    stale: bool,
}

impl Presenter {
    pub fn new(gfx: &'static Gfx, window: Arc<Window>) -> Option<Presenter> {
        let size = window.inner_size();
        let surface = match gfx.instance.create_surface(window) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("pe2: window surface: {err}");
                return None;
            }
        };
        let caps = surface.get_capabilities(&gfx.adapter);
        // The picture's values are already gamma-encoded: an sRGB target
        // would encode them again.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).or(caps.formats.first().copied())?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&gfx.device, &config);

        let device = &gfx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let picture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                picture_entry(3),
                picture_entry(4),
            ],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present params"),
            size: PARAMS as u64 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let make_pipeline = |constants: &[(&str, f64)]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("present"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = [0.0, 1.0].map(|crt| make_pipeline(&[("crt", crt)]));
        let card_pipeline = [0.0, 1.0].map(|crt| make_pipeline(&[("gain", 255.0 / 248.0), ("crt", crt)]));
        // Nearest texels when the window is an integer multiple of the
        // picture; linear smooths other sizes.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // CPU pictures have no layers: no UI marks.
        let no_ui = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("present no ui"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Some(Presenter {
            no_ui,
            gfx,
            surface,
            config,
            pipeline,
            card_pipeline,
            params,
            params_now: [0.0; PARAMS],
            card: None,
            layout,
            sampler,
            frame: None,
            pixels: Vec::new(),
            stale: false,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gfx.device, &self.config);
    }

    /// Draws the latched picture, letterboxed; false while there is none.
    pub fn present(&mut self) -> bool {
        // The PE2PC screen before the game starts (splash.rs).
        if let Some(level) = crate::splash::level()
            && let Some((picture, w, h)) = crate::splash::picture()
        {
            let scale = |c: u32| ((c as f32 * level) as u32).min(255);
            self.pixels.clear();
            self.pixels.extend(
                picture.iter().map(|&p| (scale(p >> 16 & 0xFF) << 16) | (scale(p >> 8 & 0xFF) << 8) | scale(p & 0xFF)),
            );
            self.show_pixels(*w, *h);
            return true;
        }
        if crate::runtime::gpu::front_on_card() {
            let front = crate::runtime::gpu::HW_FRONT.lock().unwrap();
            let Some(f) = front.as_ref() else { return false };
            let size = (f.width, f.height);
            if self.card.as_ref().is_none_or(|c| c.0 != size) {
                let view = |t: &wgpu::Texture| t.create_view(&Default::default());
                let bind = self.bind(&view(&f.texture), &view(&f.scene), &view(&f.ui));
                self.card = Some((size, bind));
            }
            let layered = f.layered;
            drop(front);
            let bind = self.card.as_ref().unwrap().1.clone();
            self.draw(&bind, true, size, layered);
            return true;
        }
        let Some((fw, fh)) = crate::runtime::gpu::snapshot(&mut self.pixels) else { return false };
        self.show_pixels(fw as u32, fh as u32);
        true
    }

    /// Draws `self.pixels` (`fw` x `fh` 0x00RRGGBB words), letterboxed.
    fn show_pixels(&mut self, fw: u32, fh: u32) {
        let gfx = self.gfx;
        if self.frame.as_ref().is_none_or(|f| f.1 != (fw, fh)) {
            let texture = gfx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("picture"),
                size: wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // 0x00RRGGBB words are B, G, R, 0 bytes.
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let bind = self.bind(&view, &view, &self.no_ui);
            self.frame = Some((texture, (fw, fh), bind));
        }
        let (texture, _, bind) = self.frame.as_ref().unwrap();
        gfx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck_u32(&self.pixels),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(fw * 4), rows_per_image: Some(fh) },
            wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
        );
        let bind = bind.clone();
        self.draw(&bind, false, (fw, fh), false);
    }

    fn bind(&self, view: &wgpu::TextureView, scene: &wgpu::TextureView, ui: &wgpu::TextureView) -> wgpu::BindGroup {
        self.gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("picture"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(scene) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ui) },
            ],
        })
    }

    /// Draws the picture bound by `bind`, `size` texels, letterboxed, and
    /// presents it; `layered`: its scene and UI are bound too.
    fn draw(&mut self, bind: &wgpu::BindGroup, card: bool, size: (u32, u32), layered: bool) {
        let gfx = self.gfx;
        let (mut x, mut y, mut pw, mut ph, mut whole) =
            viewport(self.config.width, self.config.height, size.0, size.1, integer_scaling());
        // Widescreen POC: fill the window and show a strip of the picture.
        let crop = widescreen_crop(self.config.width, self.config.height, size);
        if crate::runtime::trace("pan") {
            static LAST: std::sync::Mutex<(bool, bool)> = std::sync::Mutex::new((false, false));
            let now = (crop.is_some(), layered);
            let mut last = LAST.lock().unwrap();
            if *last != now {
                *last = now;
                eprintln!(
                    "[pan {:8.3}] shown: {}",
                    crate::runtime::spu::stamp(),
                    match now {
                        (false, _) => "4:3",
                        (true, false) => "widescreen, one layer",
                        (true, true) => "widescreen, scene and UI",
                    }
                );
            }
        }
        if crop.is_some() {
            (x, y, pw, ph, whole) = (0, 0, self.config.width, self.config.height, false);
        }
        let lines = crate::runtime::gpu::picture_lines().max(1) as f32;
        let (w, h) = (size.0 as f32, size.1 as f32);
        let params = match crop {
            Some((visible, top)) => [lines, 2.0, w, h, visible, top, if layered { 1.0 } else { 0.0 }, 0.0],
            None => [lines, if whole { 1.0 } else { 0.0 }, w, h, 1.0, 0.0, 0.0, 0.0],
        };
        if params != self.params_now {
            gfx.queue.write_buffer(&self.params, 0, &params.map(f32::to_le_bytes).concat());
            self.params_now = params;
        }
        if std::mem::take(&mut self.stale) {
            self.surface.configure(&gfx.device, &self.config);
        }
        let target = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.stale = true;
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&gfx.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = target.texture.create_view(&Default::default());
        let mut encoder = gfx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_viewport(x as f32, y as f32, pw as f32, ph as f32, 0.0, 1.0);
            let look = usize::from(crt());
            pass.set_pipeline(if card { &self.card_pipeline[look] } else { &self.pipeline[look] });
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        gfx.queue.submit([encoder.finish()]);
        gfx.queue.present(target);
        static SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::gfx::started();
        }
    }
}

/// The shader's `Params`, in floats.
const PARAMS: usize = 8;

/// Where the picture goes in a `ww` x `wh` window: x, y, width, height,
/// and whether it is a whole multiple of the picture's `tw` x `th` texels
/// (shown with whole texels). The picture fills the window at 4:3, or,
/// with `integer`, takes the largest whole multiple of its height that
/// fits, at 4:3 (as wide as that height wants), unless not even one does.
fn viewport(ww: u32, wh: u32, tw: u32, th: u32, integer: bool) -> (u32, u32, u32, u32, bool) {
    let fit = || if ww * 3 > wh * 4 { (wh * 4 / 3, wh) } else { (ww, ww * 3 / 4) };
    let (pw, ph) = if integer && th > 0 {
        let mut k = wh / th;
        while k > 0 && th * k * 4 / 3 > ww {
            k -= 1;
        }
        if k == 0 { fit() } else { (th * k * 4 / 3, th * k) }
    } else {
        fit()
    };
    let whole = integer && tw > 0 && th > 0 && pw % tw == 0 && ph % th == 0;
    ((ww - pw) / 2, (wh - ph) / 2, pw, ph, whole)
}
/// Aya's row on the screen as a fraction of the picture's height (NaN when
/// there is no player, infinity when she is off the picture), and a counter that rises at each camera cut. Both
/// are written by the tracker thread in `runtime/watch.rs`.
pub static AYA_FY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(f32::NAN.to_bits());
pub static AYA_CUTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// How far up and down her body reaches with that row (the top of her head
/// and her feet, fractions of the picture's height; NaN when unknown).
/// Where the strip starts now (a fraction of the picture's height), for traces.
pub static STRIP_TOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub static AYA_TOP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(f32::NAN.to_bits());
pub static AYA_BOTTOM: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(f32::NAN.to_bits());

/// The strip follows Aya instead of staying put: on unless
/// PE2_WIDESCREEN_TRACK=0 (then PE2_WIDESCREEN_PAN places it).
pub fn tracking() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PE2_WIDESCREEN_TRACK").map_or(true, |v| v.trim() != "0"))
}

struct Pan {
    target: f32,
    top: f32,
    cuts: u32,
    last: Option<std::time::Instant>,
}
static PAN_STATE: std::sync::Mutex<Pan> = std::sync::Mutex::new(Pan { target: 0.0, top: 0.0, cuts: 0, last: None });

/// Where the visible strip starts (a fraction of the picture's height): it
/// stays still while all of Aya is inside it, moves when her head or feet
/// get within a margin of its top or bottom, glides to its target, and
/// jumps at a camera cut (back to the middle, then whatever keeps her in
/// view). Taller than the strip (a close camera), her head is kept in it.
/// Without her extent, her root with fixed margins.
fn tracked_top(visible: f32) -> f32 {
    use std::sync::atomic::Ordering::{Acquire, Relaxed};
    const MARGIN_ABOVE: f32 = 0.25; // head room over her root, without her extent
    const MARGIN_BELOW: f32 = 0.10;
    // Between her extent and the strip's edges. More over her head: her
    // model is drawn up to about 7 lines (0.03) above where her joints
    // project (measured close up), and her head matters most.
    const EDGE_ABOVE: f32 = 0.06;
    const EDGE_BELOW: f32 = 0.03;
    const SETTLE: f32 = 0.12; // seconds to cover most of a move
    let max_top = 1.0 - visible;
    let centre = max_top * 0.5;
    let cuts = AYA_CUTS.load(Acquire);
    let fy = f32::from_bits(AYA_FY.load(Relaxed));
    let (head, feet) = (f32::from_bits(AYA_TOP.load(Relaxed)), f32::from_bits(AYA_BOTTOM.load(Relaxed)));
    let mut s = PAN_STATE.lock().unwrap();
    let now = std::time::Instant::now();
    let dt = s.last.map_or(0.0, |t| now.duration_since(t).as_secs_f32()).min(0.1);
    s.last = Some(now);
    let cut = cuts != s.cuts;
    s.cuts = cuts;
    if fy.is_nan() {
        s.target = centre;
    } else if fy.is_infinite() {
        // The field without Aya on the picture: hold the strip (to the
        // middle at a cut).
        if cut {
            s.target = centre;
        }
    } else {
        if cut {
            s.target = centre;
        }
        let (above, below) =
            if head <= feet { (head - EDGE_ABOVE, feet + EDGE_BELOW) } else { (fy - MARGIN_ABOVE, fy + MARGIN_BELOW) };
        if below - above > visible {
            // More of her than fits: her head, as far down as the strip goes.
            s.target = above;
        } else {
            if above < s.target {
                s.target = above;
            }
            if below > s.target + visible {
                s.target = below - visible;
            }
        }
        s.target = s.target.clamp(0.0, max_top);
    }
    if cut {
        s.top = s.target;
        if crate::runtime::trace("pan") {
            eprintln!("[pan {:8.3}] strip moved to {:.3} (Aya at {fy:.3})", crate::runtime::spu::stamp(), s.top);
        }
    } else {
        s.top += (s.target - s.top) * (1.0 - (-dt / SETTLE).exp());
    }
    s.top = s.top.clamp(0.0, max_top);
    STRIP_TOP.store(s.top.to_bits(), Relaxed);
    s.top
}
/// Widescreen. When on, a 4:3 picture in a window wider
/// than 4:3 fills the window, showing a vertical strip of it: returns the
/// strip's height as a fraction of the picture's, and where it starts (also a
/// fraction). PE2_WIDESCREEN_PAN (0 = top ... 1 = bottom, default 0.5) picks
/// the strip's place.
fn widescreen_crop(ww: u32, wh: u32, size: (u32, u32)) -> Option<(f32, f32)> {
    static PAN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    if !widescreen_on() || size.0 == 0 || size.1 * 4 != size.0 * 3 || ww * 3 <= wh * 4 {
        return None;
    }
    let pan = *PAN.get_or_init(|| {
        std::env::var("PE2_WIDESCREEN_PAN")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0)
    });
    let visible = (4.0 * wh as f32) / (3.0 * ww as f32);
    if tracking() {
        // No player on screen (title, loading): leave the picture alone.
        if f32::from_bits(AYA_FY.load(std::sync::atomic::Ordering::Relaxed)).is_nan() {
            return None;
        }
        return Some((visible, tracked_top(visible)));
    }
    Some((visible, (1.0 - visible) * pan))
}

/// Widescreen: in the field, the scene fills a window wider than 4:3 and the
/// UI is drawn over it at 4:3 (gpu's Hw::split). PE2_WIDESCREEN_POC (1 or 0),
/// else the setting ([video] widescreen; the Options' Video).
pub fn widescreen_on() -> bool {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    FORCED
        .get_or_init(|| std::env::var("PE2_WIDESCREEN_POC").ok().map(|v| v.trim() == "1"))
        .unwrap_or_else(|| crate::settings::flag(crate::settings::Id::Widescreen))
}

/// Whole multiples of the picture: PE2_INTEGER_SCALING, else the setting
/// ([video] integer_scaling; the pause menu's Options).
fn integer_scaling() -> bool {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    FORCED
        .get_or_init(|| std::env::var("PE2_INTEGER_SCALING").ok().map(|v| v.trim() != "0"))
        .unwrap_or_else(|| crate::settings::flag(crate::settings::Id::IntegerScaling))
}

/// The CRT look: PE2_CRT, else the setting ([video] crt; the pause menu's
/// Options).
fn crt() -> bool {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    FORCED
        .get_or_init(|| std::env::var("PE2_CRT").ok().map(|v| v.trim() != "0"))
        .unwrap_or_else(|| crate::settings::flag(crate::settings::Id::Crt))
}

fn bytemuck_u32(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}

#[cfg(test)]
mod tests {
    use super::viewport;

    /// The shader parses and validates (the card only reports otherwise
    /// when the game starts).
    #[test]
    fn shader_is_valid() {
        use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = wgpu::naga::front::wgsl::parse_str(super::SHADER).unwrap_or_else(|e| panic!("{e}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }

    /// Filling keeps 4:3 and centres the picture.
    #[test]
    fn fills_at_four_thirds() {
        assert_eq!(viewport(1920, 1080, 1280, 960, false), (240, 0, 1440, 1080, false));
        assert_eq!(viewport(800, 800, 320, 240, false), (0, 100, 800, 600, false));
    }

    /// Whole multiples of the picture's height, at 4:3; whole texels when
    /// the width is a multiple too; filling when not even one fits.
    #[test]
    fn integer_multiples() {
        assert_eq!(viewport(1920, 1080, 320, 240, true), (320, 60, 1280, 960, true));
        assert_eq!(viewport(1440, 1080, 1280, 960, true), (80, 60, 1280, 960, true));
        // 368 wide modes: 4:3 by height, not a multiple of the width.
        assert_eq!(viewport(1920, 1080, 368, 240, true), (320, 60, 1280, 960, false));
        // Too narrow for 4x at 1080 lines: 3x.
        assert_eq!(viewport(1200, 1080, 320, 240, true), (120, 180, 960, 720, true));
        // 960 lines do not fit 700: filling instead.
        assert_eq!(viewport(1000, 700, 1280, 960, true), (33, 0, 933, 700, false));
    }
}
