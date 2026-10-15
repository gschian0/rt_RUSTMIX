//! Control parameters. Evaluation is pure arithmetic and does not allocate.

#[derive(Clone, Copy, Debug)]
pub enum Pfield {
    Const(f32),
    /// Linear ramp across the note. `phase` is 0 at the note start and 1 at the end.
    Ramp { from: f32, to: f32 },
}

impl Pfield {
    pub fn constant(v: f32) -> Self {
        Pfield::Const(v)
    }

    #[inline]
    pub fn at(self, phase: f32) -> f32 {
        match self {
            Pfield::Const(v) => v,
            Pfield::Ramp { from, to } => from + (to - from) * phase.clamp(0.0, 1.0),
        }
    }
}

impl From<f32> for Pfield {
    fn from(v: f32) -> Self {
        Pfield::Const(v)
    }
}
