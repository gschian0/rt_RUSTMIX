//! egui score buffer, transport, meters, and score-generated synth controls.

mod gpu_visual;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use rtmix_core::{
    Engine, Shared, SCOPE_LEN, SYNTH_CHANT, SYNTH_FOF, SYNTH_FORMANT, SYNTH_LPC, SYNTH_SPEAK,
};
use rtmix_gfx::{
    load_ply, load_splat_subset, score_seconds, splat_count, Analyzer, Camera, Reactivity, Spectrum,
};
use rtmix_host::{AudioDriver, HostError};
use rtmix_minc::execute;

const LOOP_SECONDS: f64 = 16.0;
const DEFAULT_SPEECH: &str = "the voice remembers a city made of glass / we are still here / listen to the machine learn to speak";
const DEFAULT_SPLAT_BUDGET: usize = 1_200_000;

/// Scenes shipped in `assets/`, newest 3DGS captures first.
const SCENES: &[(&str, &str)] = &[
    ("garden", "assets/garden.splat"),
    ("bicycle", "assets/bicycle.splat"),
    ("treehill", "assets/treehill.splat"),
    ("room", "assets/room.splat"),
    ("truck", "assets/truck.splat"),
    ("train", "assets/train.splat"),
    ("plush", "assets/plush.splat"),
    ("stump", "assets/stump.splat"),
    ("nike", "assets/nike.splat"),
];

fn scene_label(path: &str) -> &str {
    SCENES
        .iter()
        .find(|(_, p)| *p == path)
        .map(|(label, _)| *label)
        .unwrap_or("custom")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScoreMode {
    Constellation,
    Chatter,
    Ritual,
    Pulse,
    Prism,
    Speech,
    WhisperSwarm,
    RobotOracle,
    BrokenChoir,
    WavetableLab,
    FofGalaxy,
    PhoneMatrix,
    SpatialCyclone,
    GranularBloom,
}

impl ScoreMode {
    fn label(self) -> &'static str {
        match self {
            ScoreMode::Constellation => "Constellation",
            ScoreMode::Chatter => "Lansky Chatter",
            ScoreMode::Ritual => "Ritual",
            ScoreMode::Pulse => "Pulse",
            ScoreMode::Prism => "Prism",
            ScoreMode::Speech => "Speech",
            ScoreMode::WhisperSwarm => "Whisper Swarm",
            ScoreMode::RobotOracle => "Robot Oracle",
            ScoreMode::BrokenChoir => "Broken Choir",
            ScoreMode::WavetableLab => "Wavetable Lab",
            ScoreMode::FofGalaxy => "FOF Galaxy",
            ScoreMode::PhoneMatrix => "Phone Matrix",
            ScoreMode::SpatialCyclone => "Spatial Cyclone",
            ScoreMode::GranularBloom => "Granular Bloom",
        }
    }

    fn all() -> [ScoreMode; 14] {
        [
            ScoreMode::Constellation,
            ScoreMode::Chatter,
            ScoreMode::Ritual,
            ScoreMode::Pulse,
            ScoreMode::Prism,
            ScoreMode::Speech,
            ScoreMode::WhisperSwarm,
            ScoreMode::RobotOracle,
            ScoreMode::BrokenChoir,
            ScoreMode::WavetableLab,
            ScoreMode::FofGalaxy,
            ScoreMode::PhoneMatrix,
            ScoreMode::SpatialCyclone,
            ScoreMode::GranularBloom,
        ]
    }

    fn is_speech(self) -> bool {
        matches!(
            self,
            ScoreMode::Speech
                | ScoreMode::WhisperSwarm
                | ScoreMode::RobotOracle
                | ScoreMode::BrokenChoir
        )
    }

    fn is_classic_patch(self) -> bool {
        matches!(
            self,
            ScoreMode::WavetableLab
                | ScoreMode::FofGalaxy
                | ScoreMode::PhoneMatrix
                | ScoreMode::SpatialCyclone
                | ScoreMode::GranularBloom
        )
    }

    fn description(self) -> &'static str {
        match self {
            ScoreMode::Constellation => "Balanced tonal FOF, chant, and LPC-style voices.",
            ScoreMode::Chatter => {
                "Overlapping short vocal cells with moving pitch and vowel color."
            }
            ScoreMode::Ritual => "Slow low-register pulses and sparse ceremonial rhythm.",
            ScoreMode::Pulse => "Fast Euclidean patterns with tight pitched formant bursts.",
            ScoreMode::Prism => "Bright upper formants and animated spectral motion.",
            ScoreMode::Speech => "Articulated formant speech: type words and the choir says them.",
            ScoreMode::WhisperSwarm => {
                "Breathy high-grit speech fragments swarm across the stereo field."
            }
            ScoreMode::RobotOracle => {
                "Low quantized voices answer through FOF and synthetic LPC machinery."
            }
            ScoreMode::BrokenChoir => {
                "Wide octave choirs fracture typed speech into unstable harmonies."
            }
            ScoreMode::WavetableLab => {
                "RTcmix-style wavetable lines through resonant filters, delay, and reverb."
            }
            ScoreMode::FofGalaxy => {
                "Dense FOF formant grains, chant clouds, filtering, and long echoes."
            }
            ScoreMode::PhoneMatrix => {
                "Synthetic LPC phone bursts crosscut with speech, bandpass, and feedback delay."
            }
            ScoreMode::SpatialCyclone => {
                "Wavetable and FOF sources rotate through the HOA spatial effects chain."
            }
            ScoreMode::GranularBloom => {
                "Musical micro-grains quantized to a scale, with FOF shimmer and delay."
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisualMode {
    Splat,
    GaussianReaction3D,
    AcidDrip,
    NeuralInk,
    SpectralLava,
}

impl VisualMode {
    const ALL: [Self; 5] = [
        Self::Splat,
        Self::GaussianReaction3D,
        Self::AcidDrip,
        Self::NeuralInk,
        Self::SpectralLava,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Splat => "Gaussian Splat (GPU)",
            Self::GaussianReaction3D => "Gaussian Gray-Scott (3D GPU)",
            Self::AcidDrip => "Differential Acid",
            Self::NeuralInk => "Neural Ink",
            Self::SpectralLava => "Spectral Lava",
        }
    }

    fn drip_mode(self) -> Option<u32> {
        match self {
            Self::Splat | Self::GaussianReaction3D => None,
            Self::AcidDrip => Some(0),
            Self::NeuralInk => Some(1),
            Self::SpectralLava => Some(2),
        }
    }
}

struct SynthUi {
    kind: usize,
    name: &'static str,
    gain: f32,
    tone: f32,
    motion: f32,
}

pub struct RustmixApp {
    score: String,
    status: String,
    synths: Vec<SynthUi>,
    generation: u32,
    auto_evolve: bool,
    mode: ScoreMode,
    speech_text: String,
    playing: bool,
    driver: Option<AudioDriver>,
    shared: Option<Arc<Shared>>,
    stop: Arc<AtomicBool>,
    offline: Option<thread::JoinHandle<()>>,
    splat_loaded: bool,
    splat_path: String,
    splat_status: String,
    splat_budget: usize,
    camera: Camera,
    reactivity: Reactivity,
    auto_orbit: bool,
    analyzer: Analyzer,
    spectrum: Spectrum,
    scope: Vec<f32>,
    visual_mode: VisualMode,
    visual_time: f32,
    gpu: Option<eframe::egui_wgpu::RenderState>,
}

impl RustmixApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gpu = gpu_visual::initialize(cc);
        let mut app = Self {
            score: generate_score(0, ScoreMode::Constellation, DEFAULT_SPEECH),
            status: "ready".into(),
            synths: Vec::new(),
            generation: 0,
            auto_evolve: true,
            mode: ScoreMode::Constellation,
            speech_text: DEFAULT_SPEECH.into(),
            playing: false,
            driver: None,
            shared: None,
            stop: Arc::new(AtomicBool::new(false)),
            offline: None,
            splat_loaded: false,
            splat_path: "assets/train.splat".into(),
            splat_status: "no splat loaded".into(),
            splat_budget: DEFAULT_SPLAT_BUDGET,
            camera: Camera::default(),
            reactivity: Reactivity::default(),
            auto_orbit: true,
            analyzer: Analyzer::new(SCOPE_LEN),
            spectrum: Spectrum::default(),
            scope: vec![0.0; SCOPE_LEN],
            visual_mode: VisualMode::Splat,
            visual_time: 0.0,
            gpu,
        };
        if std::path::Path::new(&app.splat_path).exists() {
            app.load_splat_asset();
        }
        app
    }

    /// Accepts `.splat` and 3DGS `.ply`, chosen by extension.
    fn load_splat_asset(&mut self) {
        let path = self.splat_path.trim().to_string();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.splat_status = format!("cannot read {path}: {err}");
                return;
            }
        };
        let lowercase = path.to_ascii_lowercase();
        // Multi-million gaussian scenes are decoded straight into the budget
        // instead of materializing the whole file first.
        let (gaussians, total) = if lowercase.ends_with(".ply") {
            match load_ply(&bytes) {
                Ok(all) => {
                    let total = all.len();
                    (Ok(all), total)
                }
                Err(err) => (Err(err.message), 0),
            }
        } else {
            let total = splat_count(&bytes).unwrap_or(0);
            (
                load_splat_subset(&bytes, self.splat_budget).map_err(|e| e.message),
                total,
            )
        };
        match gaussians {
            Ok(gaussians) if gaussians.is_empty() => {
                self.splat_status = "file parsed but held no gaussians".into();
            }
            Ok(gaussians) => {
                let (center, radius) = gpu_visual::scene_bounds(&gaussians);
                self.camera = Camera {
                    target: center,
                    distance: radius * 1.45,
                    yaw: 0.6,
                    pitch: 0.22,
                    fov_y: 0.9,
                };
                if let Some(gpu) = &self.gpu {
                    gpu_visual::upload_splats(
                        gpu,
                        &gaussians,
                        center,
                        radius,
                        total.max(gaussians.len()),
                    );
                }
                self.splat_status = format!(
                    "{} GPU splats of {}",
                    gaussians.len(),
                    total.max(gaussians.len())
                );
                self.splat_loaded = true;
            }
            Err(message) => {
                self.splat_status = format!("parse failed: {message}");
            }
        }
    }

    fn stop_playback(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.driver = None;
        if let Some(handle) = self.offline.take() {
            let _ = handle.join();
        }
        self.playing = false;
        self.stop = Arc::new(AtomicBool::new(false));
    }

    fn play(&mut self) {
        self.stop_playback();
        let mut engine = Engine::new(48_000, 2, 256);
        match execute(&self.score, &mut engine) {
            Ok(assets) => {
                let shared = engine.shared();
                self.synths = synth_controls_for_score(&self.score);
                for synth in &self.synths {
                    shared
                        .synth_controls
                        .set(synth.kind, synth.gain, synth.tone, synth.motion);
                }
                match AudioDriver::start(engine) {
                    Ok(driver) => {
                        self.status = if assets.is_empty() {
                            format!("generation {} playing", self.generation + 1)
                        } else {
                            format!(
                                "generation {} playing, {} scene file(s)",
                                self.generation + 1,
                                assets.len()
                            )
                        };
                        self.shared = Some(shared);
                        self.driver = Some(driver);
                        self.playing = true;
                    }
                    Err((HostError { message }, engine)) => {
                        self.status = format!("no device ({message}); offline clock");
                        let stop = Arc::clone(&self.stop);
                        self.shared = Some(shared);
                        self.offline = Some(thread::spawn(move || {
                            let mut engine = engine;
                            while !stop.load(Ordering::Relaxed) {
                                engine.process_block();
                                let block = engine.block_size() as f64;
                                let rate = engine.sample_rate() as f64;
                                thread::sleep(Duration::from_secs_f64(block / rate));
                            }
                        }));
                        self.playing = true;
                    }
                }
            }
            Err(err) => {
                self.status = format!("line {}: {}", err.line, err.message);
            }
        }
    }
}

impl eframe::App for RustmixApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut start_next_generation = false;
        let delta = ctx.input(|i| i.stable_dt).clamp(0.001, 0.1);
        self.visual_time += delta;
        if let Some(shared) = &self.shared {
            // Analyse the audio tap, then let the spectrum drive the picture.
            shared.scope.read(&mut self.scope);
            let rate = shared.snapshot.read().sample_rate.max(1) as f32;
            self.spectrum = self.analyzer.analyze(&self.scope, rate);
            let snap = shared.snapshot.read();
            let seconds = snap.sample_time as f32 / snap.sample_rate.max(1) as f32;
            if self.auto_evolve {
                for (index, synth) in self.synths.iter_mut().enumerate() {
                    let phase =
                        seconds * (0.13 + index as f32 * 0.037) + self.generation as f32 * 0.71;
                    synth.gain = 0.94 + 0.14 * phase.sin();
                    synth.tone = 1.0 + 0.34 * (phase * 0.73 + index as f32).sin();
                    synth.motion = 1.0 + 0.62 * (phase * 1.17 + index as f32 * 0.6).sin();
                }
            }
            for synth in &self.synths {
                shared
                    .synth_controls
                    .set(synth.kind, synth.gain, synth.tone, synth.motion);
            }
            start_next_generation = self.playing && seconds as f64 >= LOOP_SECONDS;
        }
        if start_next_generation {
            self.generation = self.generation.wrapping_add(1);
            self.score = generate_score(self.generation, self.mode, &self.speech_text);
            self.play();
        }
        let mut mode_changed = false;
        let mut speech_changed = false;
        egui::TopBottomPanel::top("transport").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if self.playing { "stop" } else { "play" })
                    .clicked()
                {
                    if self.playing {
                        self.stop_playback();
                        self.status = "stopped".into();
                    } else {
                        self.play();
                    }
                }
                ui.checkbox(&mut self.auto_evolve, "auto evolve");
                let previous_mode = self.mode;
                egui::ComboBox::from_label("mode")
                    .selected_text(self.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in ScoreMode::all() {
                            ui.selectable_value(&mut self.mode, mode, mode.label());
                        }
                    });
                mode_changed = self.mode != previous_mode;
                if ui.button("regen").clicked() {
                    self.score = generate_score(self.generation, self.mode, &self.speech_text);
                    if self.playing {
                        self.play();
                    }
                }
                ui.label(format!("generation {}", self.generation + 1));
                ui.label(&self.status);
            });
            ui.small(self.mode.description());
            if self.mode.is_speech() {
                ui.horizontal(|ui| {
                    ui.label("say:");
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.speech_text)
                            .desired_width(f32::INFINITY)
                            .hint_text("words to speak, separate phrases with /"),
                    );
                    if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        speech_changed = true;
                    }
                });
            }
        });
        if mode_changed || speech_changed {
            self.score = generate_score(self.generation, self.mode, &self.speech_text);
            if self.playing {
                self.play();
            }
        }
        egui::TopBottomPanel::bottom("meters").show(ctx, |ui| {
            if let Some(shared) = &self.shared {
                let snap = shared.snapshot.read();
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "t {:.3}s  splat {:.3}s",
                        snap.sample_time as f64 / snap.sample_rate.max(1) as f64,
                        score_seconds(snap.sample_time, snap.sample_rate)
                    ));
                    for (index, slot) in shared.out_peak.iter().take(2).enumerate() {
                        let peak = f32::from_bits(slot.load(Ordering::Relaxed));
                        ui.label(format!("out{index} {:.3}", peak));
                    }
                    let hoa = f32::from_bits(shared.hoa_peak[0].load(Ordering::Relaxed));
                    ui.label(format!("hoa {:.3}", hoa));
                });
            } else {
                ui.label("meters idle");
            }
            self.spectrum_bar(ui);
        });
        if !self.synths.is_empty() {
            egui::SidePanel::right("synth_controls")
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("live synths");
                    ui.label("generated from the playing score");
                    ui.separator();
                    for synth in &mut self.synths {
                        ui.group(|ui| {
                            ui.strong(synth.name);
                            ui.add(egui::Slider::new(&mut synth.gain, 0.0..=2.0).text("gain"));
                            ui.add(egui::Slider::new(&mut synth.tone, 0.35..=2.0).text("tone"));
                            ui.add(egui::Slider::new(&mut synth.motion, 0.0..=3.0).text("motion"));
                        });
                        ui.add_space(6.0);
                    }
                });
        }
        egui::SidePanel::left("score_panel")
            .default_width(430.0)
            .show(ctx, |ui| {
                ui.heading("score");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.score)
                            .code_editor()
                            .desired_rows(24)
                            .desired_width(f32::INFINITY),
                    );
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.visual_panel(ui, delta);
        });
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl RustmixApp {
    fn spectrum_bar(&self, ui: &mut egui::Ui) {
        let height = 26.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        let bands = self.spectrum.bands.len();
        let width = rect.width() / bands as f32;
        for (index, value) in self.spectrum.bands.iter().enumerate() {
            let level = value.clamp(0.0, 1.0);
            let bar = egui::Rect::from_min_size(
                egui::pos2(
                    rect.left() + index as f32 * width,
                    rect.bottom() - height * level,
                ),
                egui::vec2(width - 2.0, height * level),
            );
            let warm = (index as f32 / bands as f32 * 255.0) as u8;
            painter.rect_filled(
                bar,
                1.0,
                egui::Color32::from_rgb(90 + warm / 2, 150, 255 - warm / 2),
            );
        }
    }

    fn visual_panel(&mut self, ui: &mut egui::Ui, delta: f32) {
        ui.horizontal(|ui| {
            ui.strong("visual");
            egui::ComboBox::from_id_salt("visual_mode")
                .selected_text(self.visual_mode.label())
                .show_ui(ui, |ui| {
                    for mode in VisualMode::ALL {
                        ui.selectable_value(&mut self.visual_mode, mode, mode.label());
                    }
                });
            if self.visual_mode.drip_mode().is_none() {
                if self.visual_mode == VisualMode::GaussianReaction3D {
                    ui.small("2-pass GPU Gray-Scott on Morton-local splat neighbors");
                } else {
                    ui.small("wgpu instanced Gaussian rendering");
                }
            } else {
                ui.small(format!(
                    "{:.0} fps GPU reaction + domain passes",
                    1.0 / delta
                ));
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.reactivity.zoom, 0.0..=1.0).text("zoom motion"));
            ui.add(egui::Slider::new(&mut self.reactivity.twist, 0.0..=2.0).text("twist"));
            if self.visual_mode != VisualMode::Splat {
                ui.add(
                    egui::Slider::new(&mut self.reactivity.swell, 0.0..=3.0).text("domain scale"),
                );
            }
        });
        ui.separator();
        if let Some(mode) = self.visual_mode.drip_mode() {
            self.drip_panel(ui, delta, mode);
        } else {
            self.splat_panel(ui, delta);
        }
    }

    fn drip_panel(&mut self, ui: &mut egui::Ui, _delta: f32, mode: u32) {
        let available = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(available, egui::Sense::hover());
        if self.gpu.is_some() {
            ui.painter().add(gpu_visual::drip_callback(
                rect,
                mode,
                self.spectrum,
                self.reactivity,
                self.visual_time,
            ));
        } else {
            ui.centered_and_justified(|ui| ui.label("wgpu unavailable"));
        }
    }

    fn splat_panel(&mut self, ui: &mut egui::Ui, delta: f32) {
        ui.horizontal(|ui| {
            ui.heading("splat");
            egui::ComboBox::from_id_salt("splat_scene")
                .selected_text(scene_label(&self.splat_path))
                .width(110.0)
                .show_ui(ui, |ui| {
                    for (label, path) in SCENES {
                        if ui
                            .selectable_label(self.splat_path == *path, *label)
                            .clicked()
                        {
                            self.splat_path = (*path).into();
                            self.load_splat_asset();
                        }
                    }
                });
            ui.add(
                egui::TextEdit::singleline(&mut self.splat_path)
                    .desired_width(190.0)
                    .hint_text("path to a .splat or 3DGS .ply"),
            );
            if ui.button("load").clicked() {
                self.load_splat_asset();
            }
            ui.checkbox(&mut self.auto_orbit, "orbit");
            ui.small(format!(
                "{}  {:.0} fps",
                self.splat_status,
                1.0 / delta.max(1.0e-4)
            ));
        });
        ui.horizontal(|ui| {
            let quality = ui.add(
                egui::Slider::new(&mut self.splat_budget, 50_000..=2_000_000)
                    .logarithmic(true)
                    .text("detail"),
            );
            if quality.drag_stopped() || quality.lost_focus() {
                self.load_splat_asset();
            }
            ui.add(egui::Slider::new(&mut self.reactivity.displace, 0.0..=1.5).text("displace"));
            ui.add(egui::Slider::new(&mut self.reactivity.swell, 0.0..=3.0).text("swell"));
            ui.add(egui::Slider::new(&mut self.reactivity.brightness, 0.0..=3.0).text("glow"));
            ui.add(egui::Slider::new(&mut self.reactivity.spin, 0.0..=3.0).text("spin"));
            ui.add(egui::Slider::new(&mut self.reactivity.point_size, 0.5..=2.5).text("fill"));
            ui.add(egui::Slider::new(&mut self.reactivity.exposure, 0.5..=3.0).text("exposure"));
        });

        let available = ui.available_size();
        if !self.splat_loaded {
            ui.centered_and_justified(|ui| {
                ui.label(
                    "load a gaussian splat to see the audio drive it\n\n\
                     free scenes: https://huggingface.co/cakewalk/splat-data",
                );
            });
            return;
        }

        // Mouse drag orbits, scroll dollies.
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());
        if response.dragged() {
            let drag = response.drag_delta();
            self.camera.yaw -= drag.x * 0.006;
            self.camera.pitch = (self.camera.pitch + drag.y * 0.006).clamp(-1.4, 1.4);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.camera.distance =
                    (self.camera.distance * (1.0 - scroll * 0.0015)).clamp(0.05, 500.0);
            }
        }
        if self.auto_orbit {
            // Louder passages push the camera around faster.
            let speed = 0.08 + self.spectrum.level * 0.5 * self.reactivity.spin;
            self.camera.yaw += speed * delta;
            self.camera.pitch =
                (0.22 + self.spectrum.bands[1] * 0.35 * self.reactivity.spin).clamp(-1.3, 1.3);
        }

        if self.gpu.is_some() {
            ui.painter().add(gpu_visual::splat_callback(
                rect,
                self.camera,
                self.spectrum,
                self.reactivity,
                self.visual_time,
            ));
        }
    }
}

fn synth_controls_for_score(score: &str) -> Vec<SynthUi> {
    let mut synths = Vec::new();
    let choices = [
        ("FORMANT(", SYNTH_FORMANT, "FORMANT"),
        ("CHANT(", SYNTH_CHANT, "CHANT"),
        ("FOF(", SYNTH_FOF, "FOF"),
        ("LPC(", SYNTH_LPC, "LPC"),
        ("SPEAK(", SYNTH_SPEAK, "SPEAK"),
    ];
    for (call, kind, name) in choices {
        if score.lines().any(|line| {
            let code = line.split("//").next().unwrap_or("").trim_start();
            code.starts_with(call)
        }) {
            synths.push(SynthUi {
                kind,
                name,
                gain: 1.0,
                tone: 1.0,
                motion: 1.0,
            });
        }
    }
    synths
}

fn generate_score(generation: u32, mode: ScoreMode, text: &str) -> String {
    let mut state = generation
        .wrapping_mul(747_796_405)
        .wrapping_add(2_891_336_453);
    if mode.is_speech() {
        return generate_speech_score(generation, mode, text, &mut state);
    }
    if mode.is_classic_patch() {
        return generate_classic_patch(generation, mode, &mut state);
    }
    let (
        root,
        upper,
        chant_rate,
        pulses,
        steps,
        vowels,
        formant_a,
        formant_b,
        formant_c,
        reverb_mix,
        mix_gain,
    ) = match mode {
        ScoreMode::Constellation
        | ScoreMode::Speech
        | ScoreMode::WhisperSwarm
        | ScoreMode::RobotOracle
        | ScoreMode::BrokenChoir
        | ScoreMode::WavetableLab
        | ScoreMode::FofGalaxy
        | ScoreMode::PhoneMatrix
        | ScoreMode::SpatialCyclone
        | ScoreMode::GranularBloom => {
            let root = [55.0, 61.74, 73.42, 82.41, 98.0][(next_random(&mut state) as usize) % 5];
            let upper = root * [1.25, 4.0 / 3.0, 1.5, 1.75][(next_random(&mut state) as usize) % 4];
            (
                root,
                upper,
                4.0 + (next_random(&mut state) % 35) as f32 / 10.0,
                5,
                8,
                ["aeiou", "uoaei", "aioue", "eouia"][(next_random(&mut state) as usize) % 4],
                360.0 + (next_random(&mut state) % 280) as f32,
                720.0 + (next_random(&mut state) % 420) as f32,
                1150.0 + (next_random(&mut state) % 650) as f32,
                2200.0,
                0.72,
            )
        }
        ScoreMode::Chatter => {
            let root = [110.0, 146.83, 164.81, 196.0][(next_random(&mut state) as usize) % 4];
            let upper = root * [1.5, 2.0, 1.25][(next_random(&mut state) as usize) % 3];
            (
                root,
                upper,
                5.5 + (next_random(&mut state) % 25) as f32 / 10.0,
                7,
                10,
                "aeiou",
                420.0 + (next_random(&mut state) % 230) as f32,
                980.0 + (next_random(&mut state) % 520) as f32,
                1680.0 + (next_random(&mut state) % 680) as f32,
                2600.0,
                0.8,
            )
        }
        ScoreMode::Ritual => {
            let root = [41.2, 55.0, 65.41, 73.42][(next_random(&mut state) as usize) % 4];
            let upper = root * [1.25, 1.5, 2.0][(next_random(&mut state) as usize) % 3];
            (
                root,
                upper,
                3.2 + (next_random(&mut state) % 18) as f32 / 10.0,
                3,
                8,
                "uoaei",
                300.0 + (next_random(&mut state) % 180) as f32,
                650.0 + (next_random(&mut state) % 380) as f32,
                1100.0 + (next_random(&mut state) % 620) as f32,
                2800.0,
                0.68,
            )
        }
        ScoreMode::Pulse => {
            let root = [98.0, 110.0, 123.47, 146.83][(next_random(&mut state) as usize) % 4];
            let upper = root * [1.25, 1.5, 2.0, 2.25][(next_random(&mut state) as usize) % 4];
            (
                root,
                upper,
                7.0 + (next_random(&mut state) % 30) as f32 / 10.0,
                9,
                16,
                "aioue",
                480.0 + (next_random(&mut state) % 280) as f32,
                920.0 + (next_random(&mut state) % 500) as f32,
                1820.0 + (next_random(&mut state) % 780) as f32,
                2000.0,
                0.86,
            )
        }
        ScoreMode::Prism => {
            let root = [70.0, 87.31, 98.0, 116.54][(next_random(&mut state) as usize) % 4];
            let upper = root * [1.33, 1.5, 1.67, 2.0][(next_random(&mut state) as usize) % 4];
            (
                root,
                upper,
                6.2 + (next_random(&mut state) % 28) as f32 / 10.0,
                11,
                16,
                "eouia",
                520.0 + (next_random(&mut state) % 300) as f32,
                1200.0 + (next_random(&mut state) % 620) as f32,
                2200.0 + (next_random(&mut state) % 860) as f32,
                2400.0,
                0.9,
            )
        }
    };

    let (formant_amp, chant_amp) = match mode {
        ScoreMode::Chatter => (220, 650),
        ScoreMode::Pulse => (420, 1100),
        ScoreMode::Prism => (360, 900),
        ScoreMode::Ritual => (520, 1250),
        ScoreMode::Constellation
        | ScoreMode::Speech
        | ScoreMode::WhisperSwarm
        | ScoreMode::RobotOracle
        | ScoreMode::BrokenChoir
        | ScoreMode::WavetableLab
        | ScoreMode::FofGalaxy
        | ScoreMode::PhoneMatrix
        | ScoreMode::SpatialCyclone
        | ScoreMode::GranularBloom => (650, 1700),
    };
    let seed_a = next_random(&mut state);
    let seed_b = next_random(&mut state);
    let seed_c = next_random(&mut state);
    let note_label = mode.label();

    let motif = match mode {
        ScoreMode::Constellation
        | ScoreMode::Speech
        | ScoreMode::WhisperSwarm
        | ScoreMode::RobotOracle
        | ScoreMode::BrokenChoir
        | ScoreMode::WavetableLab
        | ScoreMode::FofGalaxy
        | ScoreMode::PhoneMatrix
        | ScoreMode::SpatialCyclone
        | ScoreMode::GranularBloom => String::new(),
        ScoreMode::Chatter => format!(
            r#"CHANT(0.0, 2.8, 2600, {root:.2}, 0.25, {chant_rate:.2}, 3, 8, "aei", harm)
CHANT(2.1, 3.4, 2300, {upper:.2}, 0.75, {faster:.2}, 5, 8, "oui", harm)
CHANT(5.0, 2.5, 2500, {root:.2}, 0.35, {chant_rate:.2}, 3, 8, "eoa", harm)
CHANT(7.1, 4.3, 2200, {upper:.2}, 0.65, {faster:.2}, 5, 8, "iau", harm)
LPC(0.4, 2.2, 1900, {root:.2}, 0.18, {seed_a})
LPC(2.8, 2.0, 1800, {upper:.2}, 0.82, {seed_b})
LPC(5.6, 2.4, 2000, {root:.2}, 0.28, {seed_c})
"#,
            faster = chant_rate * 1.7,
        ),
        ScoreMode::Ritual => format!(
            r#"FOF(0.0, 1.8, 1200, {root:.2}, {formant_a:.0}, 110, 0.32, 0.18, {seed_a})
FOF(2.3, 2.4, 1250, {upper:.2}, {formant_b:.0}, 96, 0.28, 0.82, {seed_b})
FOF(5.7, 1.6, 1100, {root:.2}, {formant_c:.0}, 140, 0.24, 0.35, {seed_c})
DELAY(0, 16, 1700, 0.43, 0.68)
"#,
        ),
        ScoreMode::Pulse => format!(
            r#"FOF(0.0, 0.42, 1100, {root:.2}, {formant_a:.0}, 46, 0.12, 0.1, {seed_a})
FOF(0.65, 0.38, 1000, {upper:.2}, {formant_b:.0}, 42, 0.16, 0.9, {seed_b})
FOF(1.4, 0.5, 1150, {double_root:.2}, {formant_c:.0}, 58, 0.1, 0.3, {seed_c})
FOF(2.2, 0.36, 950, {root:.2}, {formant_b:.0}, 40, 0.2, 0.72, {seed_a})
FOF(3.0, 0.55, 1200, {upper:.2}, {formant_a:.0}, 62, 0.13, 0.45, {seed_b})
DELAY(0, 16, 1300, 0.29, 0.72)
"#,
            double_root = root * 2.0,
        ),
        ScoreMode::Prism => format!(
            r#"FILTER(0, 16, 1500, {filter_cutoff:.0}, 0.7, "bandpass")
DELAY(0, 16, 1100, 0.61, 0.74)
FOF(1.0, 5.0, 900, {root:.2}, {formant_c:.0}, 160, 0.4, 0.08, {seed_a})
FOF(2.7, 4.0, 880, {upper:.2}, {formant_c:.0}, 180, 0.35, 0.92, {seed_b})
"#,
            filter_cutoff = formant_c * 0.9,
        ),
    };

    format!(
        r#"rtsetparams(48000, 2, 256)
// {note_label} generation {generation}
bus_config("FORMANT", "aux")
bus_config("CHANT", "aux")
bus_config("FOF", "out")
bus_config("LPC", "out")
bus_config("MIX", "out")
bus_config("REVERB", "out")
bus_config("DELAY", "out")
bus_config("FILTER", "out")
harm = makeharm(2048, 24, 0.85, "odd")
FORMANT(0, 16, {formant_amp}, {root:.2}, 0.5, "u", harm)
CHANT(0, 16, {chant_amp}, {upper:.2}, 0.5, {chant_rate:.2}, {pulses}, {steps}, "{vowels}", harm)
FOF(0, 15.5, 980, {root:.2}, {formant_a:.0}, 76, 0.18, 0.12, {seed_a})
FOF(0.25, 15.25, 880, {upper:.2}, {formant_b:.0}, 98, 0.16, 0.88, {seed_b})
FOF(0.5, 15, 720, {double_root:.2}, {formant_c:.0}, 132, 0.14, 0.38, {seed_c})
LPC(1, 11, 2800, {root:.2}, 0.22, {seed_b})
LPC(5, 9, 2400, {upper:.2}, 0.78, {seed_c})
MIX(0, 16, {mix_gain})
REVERB(0, 16, {reverb_mix}, 0.78, 0.28)
{motif}
"#,
        motif = motif,
        double_root = root * 2.0,
    )
}

fn generate_classic_patch(generation: u32, mode: ScoreMode, state: &mut u32) -> String {
    let root = [55.0, 65.41, 73.42, 82.41, 98.0][(next_random(state) as usize) % 5];
    let seed_a = next_random(state) % 100_000;
    let seed_b = next_random(state) % 100_000;
    let seed_c = next_random(state) % 100_000;
    let body = match mode {
        ScoreMode::WavetableLab => format!(
            r#"bus_config("WAVETABLE", "aux")
bus_config("FILTER", "aux")
bus_config("DELAY", "aux")
bus_config("REVERB", "out")
bus_config("MIX", "out")
odd = makeharm(4096, 32, 0.82, "odd")
even = makeharm(4096, 28, 0.76, "even")
WAVETABLE(0.0, 7.8, 1200, {root:.2}, 0.08, odd)
WAVETABLE(0.7, 7.2, 1050, {fifth:.2}, 0.92, even)
WAVETABLE(4.0, 8.0, 980, {upper:.2}, 0.25, odd)
WAVETABLE(7.5, 7.8, 900, {minor:.2}, 0.75, even)
FILTER(0, 16, 1100, {cutoff:.0}, 1.8, "bandpass")
DELAY(0, 16, 780, 0.293, 0.72)
REVERB(0, 16, 1250, 0.88, 0.22)
MIX(0, 16, 0.62)
"#,
            fifth = root * 1.5,
            upper = root * 2.0,
            minor = root * 1.2,
            cutoff = 620.0 + (next_random(state) % 1800) as f32,
        ),
        ScoreMode::FofGalaxy => format!(
            r#"bus_config("FOF", "aux")
bus_config("CHANT", "aux")
bus_config("FILTER", "aux")
bus_config("DELAY", "aux")
bus_config("REVERB", "out")
bus_config("MIX", "out")
harm = makeharm(2048, 30, 0.84, "odd")
FOF(0.0, 15.8, 1250, {root:.2}, 430, 62, 0.20, 0.08, {seed_a})
FOF(0.2, 15.5, 1120, {fifth:.2}, 1180, 90, 0.17, 0.92, {seed_b})
FOF(0.5, 15.0, 980, {upper:.2}, 2380, 155, 0.14, 0.42, {seed_c})
CHANT(0, 16, 1150, {chant:.2}, 0.5, 7.3, 9, 16, "aeiou", harm)
FILTER(0, 16, 850, 1750, 2.4, "bandpass")
DELAY(0, 16, 920, 0.487, 0.78)
REVERB(0, 16, 1450, 0.93, 0.18)
MIX(0, 16, 0.58)
"#,
            fifth = root * 1.5,
            upper = root * 2.01,
            chant = root * 0.75,
        ),
        ScoreMode::PhoneMatrix => format!(
            r#"bus_config("LPC", "aux")
bus_config("SPEAK", "aux")
bus_config("FOF", "aux")
bus_config("FILTER", "aux")
bus_config("DELAY", "aux")
bus_config("REVERB", "out")
bus_config("MIX", "out")
LPC(0.0, 5.2, 2600, {root:.2}, 0.12, {seed_a})
LPC(3.1, 6.0, 2300, {fifth:.2}, 0.88, {seed_b})
LPC(8.2, 6.8, 2500, {low:.2}, 0.25, {seed_c})
SPEAK(1.0, 5.5, 2100, 92, "signal lost inside the machine", 0.78, 0.15, {seed_b})
SPEAK(7.0, 6.2, 1900, 138, "answer from the other circuit", 0.62, 0.85, {seed_c})
FOF(0, 16, 620, {root:.2}, 760, 115, 0.16, 0.5, {seed_a})
FILTER(0, 16, 1050, 1850, 3.2, "bandpass")
DELAY(0, 16, 1100, 0.173, 0.81)
REVERB(0, 16, 900, 0.67, 0.42)
MIX(0, 16, 0.62)
"#,
            fifth = root * 1.5,
            low = root * 0.5,
        ),
        ScoreMode::SpatialCyclone => format!(
            r#"bus_config("WAVETABLE", "aux")
bus_config("FOF", "aux")
bus_config("SPAT", "spat")
bus_config("ROTATE", "spat")
bus_config("DECODE", "out")
harm = makeharm(4096, 36, 0.8, "odd")
WAVETABLE(0, 16, 1300, {root:.2}, 0.5, harm)
WAVETABLE(0.4, 15.4, 1050, {fifth:.2}, 0.5, harm)
FOF(0.2, 15.5, 1000, {upper:.2}, 920, 105, 0.18, 0.5, {seed_a})
FOF(0.8, 14.8, 880, {minor:.2}, 2100, 170, 0.13, 0.5, {seed_b})
SPAT(0, 16, 45, 18, 1.4)
ROTATE(0, 16, 110, 32, 75)
DECODE(0, 16, "stereo")
"#,
            fifth = root * 1.5,
            upper = root * 2.0,
            minor = root * 1.2,
        ),
        ScoreMode::GranularBloom => {
            let scale = [1.0, 1.12246, 1.25992, 1.33484, 1.49831, 1.68179, 1.88775];
            let mut grains = String::from(
                r#"bus_config("WAVETABLE", "aux")
bus_config("FOF", "aux")
bus_config("FILTER", "aux")
bus_config("DELAY", "aux")
bus_config("REVERB", "out")
bus_config("MIX", "out")
harm = makeharm(2048, 18, 0.78, "odd")
odd = makeharm(2048, 24, 0.72, "odd")
"#,
            );
            grains.push_str(&format!(
                "WAVETABLE(0, 16, 430, {:.2}, 0.32, harm)\n\
                 WAVETABLE(0, 16, 360, {:.2}, 0.68, odd)\n",
                root,
                root * 1.5
            ));
            for index in 0..96 {
                let beat = index as f32 * 0.166;
                let degree = [0usize, 2, 4, 1, 3, 5, 2, 6][index % 8];
                let octave = if index % 16 >= 12 { 2.0 } else { 1.0 };
                let grain_pitch = root * scale[degree] * octave;
                let pan = 0.12 + (index % 9) as f32 * 0.095;
                let amp = 820.0 + (index % 4) as f32 * 110.0;
                let table = if index % 3 == 0 { "harm" } else { "odd" };
                grains.push_str(&format!(
                    "WAVETABLE({beat:.3}, 0.29, {amp:.0}, {grain_pitch:.2}, {pan:.2}, {table})\n"
                ));
            }
            grains.push_str(&format!(
                "FOF(0.0, 15.8, 900, {:.2}, 1480, 82, 0.11, 0.5, {})\n\
                 FILTER(0, 16, 980, 2100, 1.8, \"bandpass\")\n\
                 DELAY(0, 16, 850, 0.333, 0.64)\n\
                 REVERB(0, 16, 920, 0.82, 0.24)\n\
                 MIX(0, 16, 0.84)\n",
                root * 2.0,
                seed_a
            ));
            grains
        }
        _ => unreachable!("classic patch generator called for non-patch mode"),
    };
    format!(
        "rtsetparams(48000, 2, 256)\n// {} generation {}\n{}",
        mode.label(),
        generation,
        body
    )
}

/// Lays the typed phrase out as overlapping spoken entries over a quiet pad.
/// Each loop re-voices the same words at new pitches, rates and positions.
fn generate_speech_score(generation: u32, mode: ScoreMode, text: &str, state: &mut u32) -> String {
    let fallback = "the voice remembers a city made of glass";
    let source = if text.trim().is_empty() {
        fallback
    } else {
        text.trim()
    };
    let phrases: Vec<&str> = source
        .split(|c| c == '/' || c == '\n')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let phrases = if phrases.is_empty() {
        vec![fallback]
    } else {
        phrases
    };

    let root = [55.0, 65.41, 73.42, 82.41][(next_random(state) as usize) % 4];
    let mut voices = String::new();
    let mut time = 0.0f32;
    let mut index = 0usize;
    while time < 14.5 {
        let phrase = phrases[index % phrases.len()];
        let words = phrase.split_whitespace().count().max(1) as f32;
        let dur = (words * 0.46 + 0.5).clamp(0.6, 6.0);
        let base_pitch = [86.0, 104.0, 118.0, 132.0, 156.0, 196.0]
            [(next_random(state) as usize) % 6]
            * (1.0 + (next_random(state) % 11) as f32 * 0.008);
        let pitch = match mode {
            ScoreMode::WhisperSwarm => base_pitch * 1.55,
            ScoreMode::RobotOracle => {
                [61.74, 73.42, 82.41, 98.0][(next_random(state) as usize) % 4]
            }
            ScoreMode::BrokenChoir => base_pitch * [0.5, 1.0, 2.0][index % 3],
            _ => base_pitch,
        };
        let amp = match mode {
            ScoreMode::WhisperSwarm => 1900 + (next_random(state) % 1000),
            ScoreMode::RobotOracle => 3000 + (next_random(state) % 1200),
            _ => 2600 + (next_random(state) % 1400),
        };
        let grit = match mode {
            ScoreMode::WhisperSwarm => 0.68 + (next_random(state) % 30) as f32 / 100.0,
            ScoreMode::RobotOracle => 0.78 + (next_random(state) % 20) as f32 / 100.0,
            ScoreMode::BrokenChoir => 0.35 + (next_random(state) % 55) as f32 / 100.0,
            _ => 0.15 + (next_random(state) % 70) as f32 / 100.0,
        };
        let pan = (next_random(state) % 101) as f32 / 100.0;
        let seed = next_random(state) % 100_000;
        voices.push_str(&format!(
            "SPEAK({time:.2}, {dur:.2}, {amp}, {pitch:.2}, \"{phrase}\", {grit:.2}, {pan:.2}, {seed})\n"
        ));

        let layer_count = match mode {
            ScoreMode::WhisperSwarm => 5 + (next_random(state) % 3) as usize,
            ScoreMode::RobotOracle => 2 + (next_random(state) % 2) as usize,
            ScoreMode::BrokenChoir => 4 + (next_random(state) % 3) as usize,
            _ => 2 + (next_random(state) % 3) as usize,
        };
        for layer in 0..layer_count {
            let layer_pitch =
                pitch * [0.5, 0.667, 0.8, 1.25, 1.5, 2.0][(next_random(state) as usize) % 6];
            let offset = time + (layer as f32 * 0.14) + (next_random(state) % 40) as f32 / 100.0;
            let layer_amp = amp / ((2 + layer) as u32).max(1);
            let layer_grit =
                (0.25 + (next_random(state) % 55) as f32 / 100.0) + layer as f32 * 0.08;
            let layer_pan =
                (pan + (layer as f32 * 0.22) + (next_random(state) % 21) as f32 / 100.0)
                    .clamp(0.0, 1.0);
            let layer_seed = next_random(state) % 100_000;
            voices.push_str(&format!(
                "SPEAK({offset:.2}, {dur:.2}, {layer_amp}, {layer_pitch:.2}, \"{phrase}\", {layer_grit:.2}, {layer_pan:.2}, {layer_seed})\n"
            ));
            if layer % 2 == 0 {
                let sub_pitch = root * [1.0, 1.5, 2.0, 3.0][(next_random(state) as usize) % 4];
                let sub_offset = offset + 0.05;
                let sub_seed = next_random(state) % 100_000;
                voices.push_str(&format!(
                    "FOF({sub_offset:.2}, {dur:.2}, {sub_amp}, {sub_pitch:.2}, {formant:.0}, 90, 0.2, 0.7, {sub_seed})\n",
                    sub_amp = (amp / 4) as u32,
                    formant = 420.0 + (next_random(state) % 260) as f32,
                ));
                let pad_seed = next_random(state) % 100_000;
                voices.push_str(&format!(
                    "LPC({sub_offset:.2}, {dur:.2}, {pad_amp}, {pad_pitch:.2}, 0.32, {pad_seed})\n",
                    pad_amp = (amp / 3) as u32,
                    pad_pitch = pitch * 0.75,
                ));
            }
        }

        if index % 2 == 0 {
            voices.push_str(&format!(
                "FORMANT({time:.2}, {dur:.2}, {shimmer_amp}, {shimmer_pitch:.2}, 0.35, \"a\", harm)\n",
                shimmer_amp = 520 + (next_random(state) % 680),
                shimmer_pitch = pitch * [0.5, 0.75, 1.5, 2.0][(next_random(state) as usize) % 4],
            ));
            voices.push_str(&format!(
                "CHANT({time:.2}, {dur:.2}, {shimmer_amp}, {shimmer_pitch:.2}, 0.18, 4.8, 6, 10, \"aeiou\", harm)\n",
                shimmer_amp = 480 + (next_random(state) % 620),
                shimmer_pitch = pitch * 0.62,
            ));
        }

        let spacing = match mode {
            ScoreMode::WhisperSwarm => 0.36,
            ScoreMode::BrokenChoir => 0.5,
            _ => 0.72,
        };
        time += dur * spacing + (next_random(state) % 60) as f32 / 100.0;
        index += 1;
    }

    format!(
        r#"rtsetparams(48000, 2, 256)
// {speech_label} generation {generation}
bus_config("FORMANT", "aux")
bus_config("SPEAK", "out")
bus_config("FOF", "out")
bus_config("MIX", "out")
bus_config("REVERB", "out")
bus_config("DELAY", "out")
harm = makeharm(2048, 20, 0.8, "odd")
FORMANT(0, 16, 300, {root:.2}, 0.5, "u", harm)
FOF(0, 16, 420, {root:.2}, {pad_formant:.0}, 120, 0.2, 0.3, {pad_seed})
{voices}MIX(0, 16, 0.7)
REVERB(0, 16, 1800, 0.72, 0.18)
DELAY(0, 16, 700, 0.37, 0.42)
"#,
        pad_formant = 420.0 + (next_random(state) % 260) as f32,
        pad_seed = next_random(state) % 100_000,
        speech_label = mode.label(),
    )
}

fn next_random(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_scores_execute() {
        for mode in ScoreMode::all() {
            for generation in 0..8 {
                let mut engine = Engine::new(48_000, 2, 256);
                execute(
                    &generate_score(generation, mode, DEFAULT_SPEECH),
                    &mut engine,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn speech_mode_emits_speak_calls_for_typed_words() {
        let score = generate_score(3, ScoreMode::Speech, "hello operator / are you there");
        assert!(score.contains("SPEAK("), "no SPEAK call:\n{score}");
        assert!(
            score.contains("\"hello operator\""),
            "phrase missing:\n{score}"
        );
        assert!(
            score.contains("\"are you there\""),
            "phrase missing:\n{score}"
        );
        let mut engine = Engine::new(48_000, 2, 256);
        execute(&score, &mut engine).unwrap();
        let controls = synth_controls_for_score(&score);
        assert!(controls.iter().any(|s| s.name == "SPEAK"));
    }

    #[test]
    fn empty_speech_text_falls_back_to_a_phrase() {
        let score = generate_score(1, ScoreMode::Speech, "   ");
        assert!(score.contains("SPEAK("));
        let mut engine = Engine::new(48_000, 2, 256);
        execute(&score, &mut engine).unwrap();
    }

    #[test]
    fn talking_modes_generate_distinct_voice_fields() {
        let text = "machines dream in liquid language";
        let clean = generate_score(2, ScoreMode::Speech, text);
        let whisper = generate_score(2, ScoreMode::WhisperSwarm, text);
        let robot = generate_score(2, ScoreMode::RobotOracle, text);
        let choir = generate_score(2, ScoreMode::BrokenChoir, text);

        assert!(whisper.contains("Whisper Swarm"));
        assert!(robot.contains("Robot Oracle"));
        assert!(choir.contains("Broken Choir"));
        assert!(whisper.matches("SPEAK(").count() > clean.matches("SPEAK(").count());
        assert_ne!(clean, robot);
        assert_ne!(robot, choir);
    }

    #[test]
    fn classic_patches_use_distinct_generator_and_effect_chains() {
        let wavetable = generate_score(4, ScoreMode::WavetableLab, DEFAULT_SPEECH);
        let fof = generate_score(4, ScoreMode::FofGalaxy, DEFAULT_SPEECH);
        let phone = generate_score(4, ScoreMode::PhoneMatrix, DEFAULT_SPEECH);
        let spatial = generate_score(4, ScoreMode::SpatialCyclone, DEFAULT_SPEECH);

        assert!(wavetable.contains("WAVETABLE(") && wavetable.contains("FILTER("));
        assert!(fof.matches("FOF(").count() >= 3 && fof.contains("REVERB("));
        assert!(phone.contains("LPC(") && phone.contains("SPEAK(") && phone.contains("DELAY("));
        assert!(
            spatial.contains("SPAT(") && spatial.contains("ROTATE(") && spatial.contains("DECODE(")
        );
        for score in [wavetable, fof, phone, spatial] {
            let mut engine = Engine::new(48_000, 2, 256);
            execute(&score, &mut engine).unwrap();
        }
    }
}

pub fn run() -> eframe::Result<()> {
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = eframe::egui_wgpu::wgpu::Backends::DX12;
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 740.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    eframe::run_native(
        "Rustmix",
        options,
        Box::new(|cc| Ok(Box::new(RustmixApp::new(cc)))),
    )
}
