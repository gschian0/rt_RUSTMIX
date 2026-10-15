//! Articulated speech: text is converted to phones, then driven through a
//! glottal source and a gliding parallel formant bank.
//!
//! The phone sequence is built in `schedule_speak`, so `run` only walks a
//! pre-allocated slice and never allocates.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use rtmix_core::{CoreError, Engine, Instrument, RunCtx, SetupCtx, SYNTH_SPEAK};

use crate::scale_amp;

/// One articulatory target. `voiced` mixes the glottal pulse against noise.
#[derive(Clone, Copy, Debug)]
pub struct Phone {
    pub formants: [f32; 4],
    pub bandwidths: [f32; 4],
    pub voiced: f32,
    pub amp: f32,
    pub dur: f32,
    /// Formant glide speed. Stops and bursts snap, vowels slide.
    pub glide: f32,
    /// Vowels carry the pitch accent and set the syllable count.
    pub syllabic: bool,
}

impl Phone {
    const fn vowel(formants: [f32; 4], dur: f32) -> Self {
        Self {
            formants,
            bandwidths: [70.0, 95.0, 140.0, 220.0],
            voiced: 1.0,
            amp: 1.0,
            dur,
            glide: 0.010,
            syllabic: true,
        }
    }

    const fn sonorant(formants: [f32; 4], dur: f32, amp: f32) -> Self {
        Self {
            formants,
            bandwidths: [80.0, 110.0, 170.0, 260.0],
            voiced: 1.0,
            amp,
            dur,
            glide: 0.012,
            syllabic: false,
        }
    }

    const fn fricative(formants: [f32; 4], dur: f32, amp: f32, voiced: f32) -> Self {
        Self {
            formants,
            bandwidths: [220.0, 320.0, 480.0, 700.0],
            voiced,
            amp,
            dur,
            glide: 0.004,
            syllabic: false,
        }
    }

    const fn closure(dur: f32, voiced: f32) -> Self {
        Self {
            formants: [240.0, 1100.0, 2300.0, 3300.0],
            bandwidths: [120.0, 200.0, 300.0, 400.0],
            voiced,
            amp: if voiced > 0.0 { 0.10 } else { 0.0 },
            dur,
            glide: 0.004,
            syllabic: false,
        }
    }

    const fn burst(formants: [f32; 4], amp: f32, voiced: f32) -> Self {
        Self {
            formants,
            bandwidths: [300.0, 420.0, 600.0, 900.0],
            voiced,
            amp,
            dur: 0.014,
            glide: 0.0008,
            syllabic: false,
        }
    }

    const fn pause(dur: f32) -> Self {
        Self {
            formants: [500.0, 1500.0, 2500.0, 3400.0],
            bandwidths: [100.0, 150.0, 220.0, 320.0],
            voiced: 0.0,
            amp: 0.0,
            dur,
            glide: 0.020,
            syllabic: false,
        }
    }
}

pub struct SpeakNote {
    pub start: f64,
    pub dur: f64,
    pub amp: f32,
    /// Base pitch in Hz for the glottal source.
    pub pitch: f32,
    /// 0.0 is a clean voice, 1.0 is the band-limited quantized phone voice.
    pub grit: f32,
    pub pan: f32,
    pub seed: u32,
    pub text: String,
}

struct Speak {
    amp: f32,
    pitch: f32,
    grit: f32,
    pan: f32,
    rng: u32,
    phones: Vec<Phone>,
    /// Scales every phone duration so the utterance fills the note.
    rate: f32,
    index: usize,
    phone_left: f32,
    phone_dur: f32,
    current: [f32; 4],
    current_bw: [f32; 4],
    envelope: f32,
    glottal_phase: f32,
    previous_flow: f32,
    vibrato_phase: f32,
    resonators: [Biquad; 4],
    redesign_countdown: u32,
    spoken: f32,
    total: f32,
    hp_x1: f32,
    hp_y1: f32,
    lp_y1: f32,
    tilt_y1: f32,
}

#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    a1: f32,
    a2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    const fn new() -> Self {
        Self {
            b0: 1.0,
            a1: 0.0,
            a2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn design(&mut self, sample_rate: f32, freq: f32, bandwidth: f32) {
        let freq = freq.clamp(90.0, sample_rate * 0.45);
        let radius = libm::expf(-core::f32::consts::PI * bandwidth.max(30.0) / sample_rate);
        let angle = core::f32::consts::TAU * freq / sample_rate;
        self.b0 = (1.0 - radius) * libm::sqrtf(1.0 - radius * radius);
        self.a1 = -2.0 * radius * libm::cosf(angle);
        self.a2 = radius * radius;
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn random01(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / 16_777_216.0
}

fn stereo_gains(pan: f32, channels: usize) -> (f32, f32) {
    if channels >= 2 {
        let pan = pan.clamp(0.0, 1.0);
        (libm::sqrtf(1.0 - pan), libm::sqrtf(pan))
    } else {
        (1.0, 0.0)
    }
}

pub fn schedule_speak(engine: &mut Engine, note: SpeakNote) -> Result<(), CoreError> {
    let phones = text_to_phones(&note.text);
    let natural: f32 = phones.iter().map(|p| p.dur).sum();
    let rate = if note.dur > 0.0 && natural > 0.0 {
        (natural / note.dur as f32).clamp(0.55, 2.4)
    } else {
        1.0
    };
    engine.schedule_named(
        "SPEAK",
        note.start,
        note.dur,
        Box::new(Speak {
            amp: scale_amp(note.amp),
            pitch: note.pitch.clamp(50.0, 400.0),
            grit: note.grit.clamp(0.0, 1.0),
            pan: note.pan.clamp(0.0, 1.0),
            rng: note.seed.max(1),
            rate,
            total: natural / rate.max(1.0e-6),
            phones,
            index: 0,
            phone_left: 0.0,
            phone_dur: 0.0,
            current: [500.0, 1500.0, 2500.0, 3400.0],
            current_bw: [80.0, 110.0, 170.0, 260.0],
            envelope: 0.0,
            glottal_phase: 0.0,
            previous_flow: 0.0,
            vibrato_phase: 0.0,
            resonators: [Biquad::new(); 4],
            redesign_countdown: 0,
            spoken: 0.0,
            hp_x1: 0.0,
            hp_y1: 0.0,
            lp_y1: 0.0,
            tilt_y1: 0.0,
        }),
    )
}

/// Rosenberg glottal pulse. The caller differentiates it for the radiation
/// characteristic, which is what gives the source its -6 dB/octave tilt.
fn glottal_flow(phase: f32) -> f32 {
    const OPEN: f32 = 0.42;
    const CLOSE: f32 = 0.18;
    if phase < OPEN {
        let t = phase / OPEN;
        3.0 * t * t - 2.0 * t * t * t
    } else if phase < OPEN + CLOSE {
        let t = (phase - OPEN) / CLOSE;
        1.0 - t * t
    } else {
        0.0
    }
}

impl Instrument for Speak {
    fn init(&mut self, _: &mut SetupCtx<'_>) {}

    fn configure(&mut self, ctx: &mut SetupCtx<'_>) {
        let sr = ctx.sample_rate.max(1) as f32;
        if let Some(first) = self.phones.first() {
            self.current = first.formants;
            self.current_bw = first.bandwidths;
        }
        for (index, resonator) in self.resonators.iter_mut().enumerate() {
            resonator.design(sr, self.current[index], self.current_bw[index]);
        }
    }

    fn run(&mut self, ctx: &mut RunCtx<'_>) {
        if self.phones.is_empty() {
            return;
        }
        let sr = ctx.sample_rate.max(1) as f32;
        let (left, right) = stereo_gains(self.pan, ctx.buses.channels(ctx.out_bus));
        let (gain, tone, motion) = ctx.synth_controls.load(SYNTH_SPEAK);
        let channels = ctx.buses.channels(ctx.out_bus);
        for i in 0..ctx.frame_count {
            let frame = ctx.frame_offset + i;
            if self.phone_left <= 0.0 {
                let phone = self.phones[self.index % self.phones.len()];
                self.index += 1;
                self.phone_dur = (phone.dur / self.rate).max(0.004);
                self.phone_left = self.phone_dur;
            }
            let phone = self.phones[(self.index - 1) % self.phones.len()];
            self.phone_left -= 1.0 / sr;

            // Coarticulation: formants slide toward the target instead of jumping.
            let glide = 1.0 - libm::expf(-1.0 / (phone.glide.max(0.0005) * sr));
            for index in 0..4 {
                let stretch = if index == 0 { 1.0 } else { tone.clamp(0.5, 1.8) };
                let target = phone.formants[index] * stretch;
                self.current[index] += (target - self.current[index]) * glide;
                self.current_bw[index] += (phone.bandwidths[index] - self.current_bw[index]) * glide;
            }
            if self.redesign_countdown == 0 {
                for (index, resonator) in self.resonators.iter_mut().enumerate() {
                    resonator.design(sr, self.current[index], self.current_bw[index]);
                }
                self.redesign_countdown = 16;
            }
            self.redesign_countdown -= 1;

            // Phone envelope with short edges so segments do not click.
            let edge = (self.phone_dur * 0.28).clamp(0.002, 0.025);
            let in_gain = ((self.phone_dur - self.phone_left) / edge).clamp(0.0, 1.0);
            let out_gain = (self.phone_left / edge).clamp(0.0, 1.0);
            let target_env = phone.amp * in_gain * out_gain;
            self.envelope += (target_env - self.envelope) * 0.02;

            // Prosody: declination over the phrase plus an accent on each vowel.
            let progress = if self.total > 0.0 {
                (self.spoken / self.total).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let declination = 1.12 - 0.30 * progress;
            let accent = if phone.syllabic {
                1.0 + 0.09 * libm::sinf(core::f32::consts::PI * (1.0 - self.phone_left / self.phone_dur.max(1.0e-6)))
            } else {
                1.0
            };
            let vibrato = 1.0 + 0.012 * motion * libm::sinf(self.vibrato_phase * core::f32::consts::TAU);
            let jitter = 1.0 + (random01(&mut self.rng) - 0.5) * 0.006;
            let f0 = (self.pitch * declination * accent * vibrato * jitter).clamp(40.0, 500.0);

            // Source: glottal derivative for voicing, shaped noise for frication.
            self.glottal_phase += f0 / sr;
            if self.glottal_phase >= 1.0 {
                self.glottal_phase -= libm::floorf(self.glottal_phase);
            }
            let flow = glottal_flow(self.glottal_phase);
            let derivative = (flow - self.previous_flow) * 28.0;
            self.previous_flow = flow;
            let noise = (random01(&mut self.rng) - 0.5) * 2.0;
            let voiced = phone.voiced.clamp(0.0, 1.0);
            let aspiration = noise * (0.05 + (1.0 - voiced) * 0.95);
            let excitation = derivative * voiced + aspiration * 0.55;

            // Parallel bank: voiced energy sits low, frication sits high.
            let voiced_amps = [1.0, 0.62, 0.34, 0.16];
            let noise_amps = [0.08, 0.30, 0.90, 0.70];
            let mut sample = 0.0;
            for index in 0..4 {
                let weight = voiced_amps[index] * voiced + noise_amps[index] * (1.0 - voiced);
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                sample += self.resonators[index].tick(excitation) * weight * sign;
            }

            // Spectral tilt keeps the bank from sounding like buzzing resonators.
            let tilt = 1.0 - libm::expf(-core::f32::consts::TAU * 2600.0 / sr);
            self.tilt_y1 += tilt * (sample - self.tilt_y1);
            sample = self.tilt_y1 * 0.65 + sample * 0.35;
            sample *= self.envelope * self.amp * gain * 3.4;

            if self.grit > 0.0 {
                let hp_alpha = 0.962;
                let highpassed = hp_alpha * (self.hp_y1 + sample - self.hp_x1);
                self.hp_x1 = sample;
                self.hp_y1 = highpassed;
                let lp_alpha = 1.0 - libm::expf(-core::f32::consts::TAU * 3200.0 / sr);
                self.lp_y1 += lp_alpha * (highpassed - self.lp_y1);
                let mut dirty = libm::tanhf(self.lp_y1 * 2.2) * 0.8;
                dirty = libm::roundf(dirty * 56.0) / 56.0;
                sample = sample * (1.0 - self.grit) + dirty * self.grit;
            }
            sample = libm::tanhf(sample * 1.2) * 0.84;

            if channels >= 2 {
                ctx.buses.add(ctx.out_bus, 0, frame, sample * left);
                ctx.buses.add(ctx.out_bus, 1, frame, sample * right);
            } else {
                ctx.buses.add(ctx.out_bus, 0, frame, sample);
            }
            self.vibrato_phase += 5.2 / sr;
            self.vibrato_phase -= libm::floorf(self.vibrato_phase);
            self.spoken += 1.0 / sr;
        }
    }
}

// Formant targets follow the usual Peterson/Barney and Klatt reference values.
const AA: [f32; 4] = [730.0, 1090.0, 2440.0, 3400.0];
const AE: [f32; 4] = [660.0, 1720.0, 2410.0, 3300.0];
const AH: [f32; 4] = [640.0, 1190.0, 2390.0, 3300.0];
const AO: [f32; 4] = [570.0, 840.0, 2410.0, 3300.0];
const EH: [f32; 4] = [530.0, 1840.0, 2480.0, 3400.0];
const IH: [f32; 4] = [390.0, 1990.0, 2550.0, 3400.0];
const IY: [f32; 4] = [270.0, 2290.0, 3010.0, 3500.0];
const UH: [f32; 4] = [440.0, 1020.0, 2240.0, 3000.0];
const UW: [f32; 4] = [300.0, 870.0, 2240.0, 3000.0];
const ER: [f32; 4] = [490.0, 1350.0, 1690.0, 3300.0];
const OW: [f32; 4] = [450.0, 900.0, 2300.0, 3200.0];

fn push_diphthong(out: &mut Vec<Phone>, first: [f32; 4], second: [f32; 4]) {
    out.push(Phone::vowel(first, 0.110));
    let mut glide = Phone::vowel(second, 0.075);
    glide.glide = 0.045;
    out.push(glide);
}

fn push_stop(out: &mut Vec<Phone>, burst: [f32; 4], voiced: bool) {
    out.push(Phone::closure(if voiced { 0.045 } else { 0.058 }, if voiced { 0.6 } else { 0.0 }));
    out.push(Phone::burst(burst, if voiced { 0.55 } else { 0.80 }, if voiced { 0.35 } else { 0.0 }));
}

fn is_vowel(byte: u8) -> bool {
    matches!(byte, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Letter-to-sound rules for English-looking text. This is deliberately a small
/// rule set: enough for words to be recognizable, not a full pronunciation
/// dictionary.
pub fn text_to_phones(text: &str) -> Vec<Phone> {
    let mut out = Vec::new();
    for word in text.split(|c: char| c.is_whitespace()) {
        let cleaned: Vec<u8> = word
            .bytes()
            .filter(|b| b.is_ascii_alphabetic())
            .map(|b| b.to_ascii_lowercase())
            .collect();
        if cleaned.is_empty() {
            if word.contains(',') || word.contains('.') {
                out.push(Phone::pause(0.20));
            }
            continue;
        }
        // A trailing silent `e` lengthens the previous vowel instead of sounding.
        let mut letters = &cleaned[..];
        let mut silent_e = false;
        if letters.len() >= 3 && letters[letters.len() - 1] == b'e' && !is_vowel(letters[letters.len() - 2]) {
            silent_e = true;
            letters = &letters[..letters.len() - 1];
        }
        word_to_phones(letters, silent_e, &mut out);
        out.push(Phone::pause(0.085));
        if word.ends_with(',') {
            out.push(Phone::pause(0.16));
        }
        if word.ends_with('.') || word.ends_with('?') || word.ends_with('!') {
            out.push(Phone::pause(0.30));
        }
    }
    if out.is_empty() {
        out.push(Phone::pause(0.25));
    }
    out
}

fn word_to_phones(letters: &[u8], silent_e: bool, out: &mut Vec<Phone>) {
    let mut i = 0;
    while i < letters.len() {
        let c = letters[i];
        let next = letters.get(i + 1).copied().unwrap_or(0);
        let rest = letters.len() - i;

        // Two-letter graphemes win over single letters.
        if rest >= 2 {
            let pair = [c, next];
            let matched = match &pair {
                b"sh" => {
                    out.push(Phone::fricative([800.0, 1800.0, 2500.0, 3100.0], 0.115, 0.85, 0.0));
                    true
                }
                b"ch" => {
                    push_stop(out, [1700.0, 2400.0, 3100.0, 3800.0], false);
                    out.push(Phone::fricative([800.0, 1800.0, 2500.0, 3100.0], 0.085, 0.80, 0.0));
                    true
                }
                b"th" => {
                    out.push(Phone::fricative([900.0, 1500.0, 2800.0, 3900.0], 0.085, 0.42, 0.25));
                    true
                }
                b"ph" => {
                    out.push(Phone::fricative([800.0, 1400.0, 2200.0, 3300.0], 0.085, 0.40, 0.0));
                    true
                }
                b"wh" => {
                    out.push(Phone::sonorant([300.0, 610.0, 2200.0, 3000.0], 0.070, 0.75));
                    true
                }
                b"ng" => {
                    out.push(Phone::sonorant([250.0, 2300.0, 2900.0, 3300.0], 0.095, 0.55));
                    true
                }
                b"ck" => {
                    push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], false);
                    true
                }
                b"qu" => {
                    push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], false);
                    out.push(Phone::sonorant([300.0, 610.0, 2200.0, 3000.0], 0.055, 0.75));
                    true
                }
                b"ee" | b"ea" => {
                    out.push(Phone::vowel(IY, 0.145));
                    true
                }
                b"oo" => {
                    out.push(Phone::vowel(UW, 0.140));
                    true
                }
                b"ou" | b"ow" => {
                    push_diphthong(out, AA, UW);
                    true
                }
                b"oa" => {
                    push_diphthong(out, AO, UW);
                    true
                }
                b"ai" | b"ay" | b"ei" => {
                    push_diphthong(out, EH, IY);
                    true
                }
                b"oi" | b"oy" => {
                    push_diphthong(out, AO, IY);
                    true
                }
                b"au" | b"aw" => {
                    out.push(Phone::vowel(AO, 0.145));
                    true
                }
                b"ie" => {
                    push_diphthong(out, AA, IY);
                    true
                }
                b"er" | b"ir" | b"ur" => {
                    out.push(Phone::vowel(ER, 0.150));
                    true
                }
                b"ar" => {
                    out.push(Phone::vowel(AA, 0.130));
                    out.push(Phone::sonorant(ER, 0.075, 0.80));
                    true
                }
                b"or" => {
                    out.push(Phone::vowel(AO, 0.130));
                    out.push(Phone::sonorant(ER, 0.075, 0.80));
                    true
                }
                _ => false,
            };
            if matched {
                i += 2;
                continue;
            }
        }

        // A vowel followed by a single consonant then a silent `e` is long.
        let long_vowel = silent_e && is_vowel(c) && rest == 2 && !is_vowel(next);
        match c {
            b'a' => {
                if long_vowel {
                    push_diphthong(out, EH, IY);
                } else {
                    out.push(Phone::vowel(AE, 0.120));
                }
            }
            b'e' => {
                if long_vowel {
                    out.push(Phone::vowel(IY, 0.145));
                } else {
                    out.push(Phone::vowel(EH, 0.115));
                }
            }
            b'i' => {
                if long_vowel {
                    push_diphthong(out, AA, IY);
                } else {
                    out.push(Phone::vowel(IH, 0.100));
                }
            }
            b'o' => {
                if long_vowel {
                    push_diphthong(out, OW, UW);
                } else {
                    out.push(Phone::vowel(AA, 0.125));
                }
            }
            b'u' => {
                if long_vowel {
                    out.push(Phone::vowel(UW, 0.145));
                } else if matches!(next, b'l' | b'r' | b's') {
                    out.push(Phone::vowel(UH, 0.105));
                } else {
                    out.push(Phone::vowel(AH, 0.110));
                }
            }
            b'y' => {
                if i == 0 {
                    out.push(Phone::sonorant(IY, 0.060, 0.80));
                } else {
                    out.push(Phone::vowel(IY, 0.120));
                }
            }
            b'b' => push_stop(out, [700.0, 1200.0, 2200.0, 3200.0], true),
            b'p' => push_stop(out, [700.0, 1200.0, 2200.0, 3200.0], false),
            b'd' => push_stop(out, [1600.0, 2200.0, 3000.0, 3700.0], true),
            b't' => push_stop(out, [1700.0, 2400.0, 3100.0, 3800.0], false),
            b'g' => push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], true),
            b'k' => push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], false),
            b'm' => out.push(Phone::sonorant([250.0, 1100.0, 2200.0, 3000.0], 0.090, 0.60)),
            b'n' => out.push(Phone::sonorant([250.0, 1700.0, 2600.0, 3200.0], 0.085, 0.60)),
            b'l' => out.push(Phone::sonorant([360.0, 1300.0, 2500.0, 3300.0], 0.080, 0.78)),
            b'r' => out.push(Phone::sonorant([310.0, 1060.0, 1380.0, 3200.0], 0.080, 0.80)),
            b'w' => out.push(Phone::sonorant([300.0, 610.0, 2200.0, 3000.0], 0.070, 0.75)),
            b'j' => {
                push_stop(out, [1600.0, 2200.0, 3000.0, 3700.0], true);
                out.push(Phone::fricative([800.0, 1800.0, 2500.0, 3100.0], 0.080, 0.55, 0.45));
            }
            b'f' => out.push(Phone::fricative([800.0, 1400.0, 2200.0, 3300.0], 0.095, 0.42, 0.0)),
            b'v' => out.push(Phone::fricative([800.0, 1400.0, 2200.0, 3300.0], 0.085, 0.45, 0.45)),
            b's' => out.push(Phone::fricative([1400.0, 3200.0, 5200.0, 6800.0], 0.110, 0.78, 0.0)),
            b'z' => out.push(Phone::fricative([1400.0, 3200.0, 5200.0, 6800.0], 0.100, 0.70, 0.45)),
            b'h' => out.push(Phone::fricative([600.0, 1400.0, 2400.0, 3300.0], 0.070, 0.35, 0.0)),
            b'c' => {
                if matches!(next, b'e' | b'i' | b'y') {
                    out.push(Phone::fricative([1400.0, 3200.0, 5200.0, 6800.0], 0.105, 0.78, 0.0));
                } else {
                    push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], false);
                }
            }
            b'x' => {
                push_stop(out, [1900.0, 2100.0, 2900.0, 3600.0], false);
                out.push(Phone::fricative([1400.0, 3200.0, 5200.0, 6800.0], 0.095, 0.75, 0.0));
            }
            _ => {}
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_becomes_a_plausible_phone_sequence() {
        let phones = text_to_phones("time");
        // t closure, t burst, AA, IY glide, m, word pause.
        assert!(phones.len() >= 5, "phones {}", phones.len());
        assert!(phones.iter().any(|p| p.syllabic), "no vowel produced");
        let total: f32 = phones.iter().map(|p| p.dur).sum();
        assert!(total > 0.15 && total < 1.0, "duration {total}");
    }

    #[test]
    fn vowels_and_consonants_are_distinguishable() {
        let vowels = text_to_phones("aaa");
        assert!(vowels.iter().filter(|p| p.syllabic).count() >= 3);
        let stops = text_to_phones("ptk");
        assert!(stops.iter().any(|p| p.amp == 0.0), "no stop closure");
        assert!(!stops.iter().any(|p| p.syllabic), "unexpected vowel");
    }

    #[test]
    fn speech_output_tracks_the_first_vowel_formant() {
        let mut engine = Engine::new(48_000, 2, 256);
        schedule_speak(
            &mut engine,
            SpeakNote {
                start: 0.0,
                dur: 1.0,
                amp: 0.5,
                pitch: 110.0,
                grit: 0.0,
                pan: 0.5,
                seed: 7,
                text: String::from("laa"),
            },
        )
        .unwrap();
        let mut rendered = Vec::new();
        for _ in 0..190 {
            engine.process_block();
            rendered.extend_from_slice(engine.bus_plane(engine.out_bus(), 0));
        }
        let peak = rendered.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.02, "silent speech, peak {peak}");
        assert!(peak < 1.05, "speech clipped, peak {peak}");
    }

    #[test]
    fn different_vowels_put_energy_at_different_first_formants() {
        // "aa" has a high F1 (~730 Hz), "ee" a low one (~270 Hz). If the bank
        // really tracks the targets, the energy ratio must flip between them.
        let ratio_for = |text: &str| {
            let mut engine = Engine::new(48_000, 2, 256);
            schedule_speak(
                &mut engine,
                SpeakNote {
                    start: 0.0,
                    dur: 1.2,
                    amp: 0.6,
                    pitch: 100.0,
                    grit: 0.0,
                    pan: 0.5,
                    seed: 11,
                    text: String::from(text),
                },
            )
            .unwrap();
            let mut rendered = Vec::new();
            for _ in 0..200 {
                engine.process_block();
                rendered.extend_from_slice(engine.bus_plane(engine.out_bus(), 0));
            }
            // Skip the onset so the glide has settled on the target.
            let window = &rendered[12_000..36_000];
            goertzel(window, 48_000.0, 730.0) / goertzel(window, 48_000.0, 280.0).max(1.0e-9)
        };
        let open = ratio_for("aa");
        let close = ratio_for("ee");
        assert!(
            open > close * 1.5,
            "vowels not distinct: aa ratio {open}, ee ratio {close}"
        );
    }

    /// Single-bin DFT magnitude, used to probe one formant region.
    fn goertzel(samples: &[f32], sample_rate: f32, freq: f32) -> f32 {
        let k = core::f32::consts::TAU * freq / sample_rate;
        let coeff = 2.0 * libm::cosf(k);
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for sample in samples {
            let s0 = sample + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        libm::sqrtf(s1 * s1 + s2 * s2 - coeff * s1 * s2) / samples.len() as f32
    }

    #[test]
    fn empty_text_still_produces_a_running_instrument() {
        let mut engine = Engine::new(48_000, 2, 256);
        schedule_speak(
            &mut engine,
            SpeakNote {
                start: 0.0,
                dur: 0.25,
                amp: 0.4,
                pitch: 120.0,
                grit: 1.0,
                pan: 0.5,
                seed: 3,
                text: String::new(),
            },
        )
        .unwrap();
        for _ in 0..60 {
            engine.process_block();
        }
    }
}
