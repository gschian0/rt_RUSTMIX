class RustmixProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.mixer = null;
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === "wasm") {
        this.mixer = data.mixer;
      } else if (data.type === "score" && this.mixer) {
        this.mixer.load_score(data.score);
      } else if (data.type === "pose" && this.mixer) {
        this.mixer.set_listener(data.yaw, data.pitch, data.roll);
      }
    };
  }

  process(_inputs, outputs) {
    const channels = outputs[0];
    if (!channels || channels.length === 0) {
      return true;
    }
    const frames = channels[0].length;
    if (this.mixer) {
      const interleaved = new Float32Array(frames * channels.length);
      this.mixer.process(interleaved);
      for (let frame = 0; frame < frames; frame++) {
        for (let ch = 0; ch < channels.length; ch++) {
          channels[ch][frame] = interleaved[frame * channels.length + ch];
        }
      }
    } else {
      for (let ch = 0; ch < channels.length; ch++) {
        channels[ch].fill(0);
      }
    }
    return true;
  }
}

registerProcessor("rustmix", RustmixProcessor);
