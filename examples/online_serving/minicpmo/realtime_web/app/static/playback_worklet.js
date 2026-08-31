class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.started = false;
    this.activeResponseId = null;
    this.initialBufferFrames = Math.round(sampleRate * 0.2);
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeFrames = Math.max(1, Math.round(sampleRate * 0.005));
    this.fadeInFrames = 0;
    this.progressIntervalFrames = Math.max(1, Math.round(sampleRate * 0.08));
    this.playedFramesByResponse = new Map();
    this.underrunFramesByResponse = new Map();
    this.nextProgressFrameByResponse = new Map();
    this.drainingResponseIds = new Set();
    this.terminalFadeFramesByResponse = new Map();
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === 'audio' && message.pcm) {
      const wasEmpty = this.queue.length === 0;
      if (!this.started && !this.activeResponseId) {
        this.activeResponseId = message.responseId || null;
      }
      if (!this.started && Number.isFinite(message.initialBufferMs)) {
        this.initialBufferFrames = Math.max(0, Math.round((sampleRate * message.initialBufferMs) / 1000));
      }
      this.queue.push({ pcm: message.pcm, responseId: message.responseId || null });
      if (!this.started && wasEmpty && !this.rebuffering) {
        this.bufferWaitFrames = this.initialBufferFrames;
      }
      return;
    }
    if (message.type === 'drain' && message.responseId) {
      const responseId = message.responseId;
      this.drainingResponseIds.add(responseId);
      this.terminalFadeFramesByResponse.set(
        responseId,
        Math.min(this.fadeFrames, this.bufferedFrames(responseId)),
      );
      if (!this.started && this.bufferedFrames() > 0) {
        this.bufferWaitFrames = 0;
        this.startPlayback();
      }
      this.notifyIfDrained(responseId);
      return;
    }
    if (message.type === 'clear') {
      this.clearState();
      return;
    }
    if (message.type === 'revoke' && message.responseId) {
      this.revokeResponse(message.responseId);
    }
  }

  clearState() {
    this.queue = [];
    this.offset = 0;
    this.started = false;
    this.activeResponseId = null;
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeInFrames = 0;
    this.playedFramesByResponse.clear();
    this.underrunFramesByResponse.clear();
    this.nextProgressFrameByResponse.clear();
    this.drainingResponseIds.clear();
    this.terminalFadeFramesByResponse.clear();
  }

  revokeResponse(responseId) {
    let revokedFrames = 0;
    const retained = [];
    let removedHead = false;
    this.queue.forEach((entry, index) => {
      if (entry.responseId !== responseId) {
        retained.push(entry);
        return;
      }
      if (index === 0) removedHead = true;
      revokedFrames += entry.pcm.length - (index === 0 ? this.offset : 0);
    });
    this.queue = retained;
    if (removedHead) this.offset = 0;
    const playedFrames = this.playedFramesByResponse.get(responseId) || 0;
    this.port.postMessage({
      type: 'playback-revoked',
      responseId,
      playedMs: Math.round((playedFrames * 1000) / sampleRate),
      revokedMs: Math.round((revokedFrames * 1000) / sampleRate),
    });
    this.playedFramesByResponse.delete(responseId);
    this.underrunFramesByResponse.delete(responseId);
    this.nextProgressFrameByResponse.delete(responseId);
    this.drainingResponseIds.delete(responseId);
    this.terminalFadeFramesByResponse.delete(responseId);
    if (this.activeResponseId === responseId) {
      this.started = false;
      this.activeResponseId = null;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.rebuffering = false;
      this.fadeInFrames = 0;
    }
  }

  bufferedFrames(responseId = null) {
    return this.queue.reduce((total, entry, index) => {
      if (responseId && entry.responseId !== responseId) return total;
      return total + entry.pcm.length - (index === 0 ? this.offset : 0);
    }, 0);
  }

  activateResponse(responseId) {
    if (this.activeResponseId === responseId) return;
    this.activeResponseId = responseId;
    this.fadeInFrames = this.fadeFrames;
    const playedFrames = this.playedFramesByResponse.get(responseId) || 0;
    if (playedFrames === 0) {
      this.port.postMessage({ type: 'playback-started', responseId });
    }
  }

  startPlayback() {
    if (this.started || this.queue.length === 0) return;
    this.started = true;
    const responseId = this.queue[0].responseId;
    if (this.activeResponseId !== responseId) {
      this.activateResponse(responseId);
      return;
    }
    this.fadeInFrames = this.fadeFrames;
    if ((this.playedFramesByResponse.get(responseId) || 0) === 0) {
      this.port.postMessage({ type: 'playback-started', responseId });
    }
  }

  notifyIfDrained(responseId = null) {
    const candidates = responseId ? [responseId] : Array.from(this.drainingResponseIds);
    candidates.forEach((candidate) => {
      if (!this.drainingResponseIds.has(candidate)) return;
      if (this.queue.some((entry) => entry.responseId === candidate)) return;
      const playedFrames = this.playedFramesByResponse.get(candidate) || 0;
      const underrunFrames = this.underrunFramesByResponse.get(candidate) || 0;
      this.port.postMessage({
        type: 'playback-drained',
        responseId: candidate,
        playedMs: Math.round((playedFrames * 1000) / sampleRate),
        underrunMs: Math.round((underrunFrames * 1000) / sampleRate),
      });
      this.playedFramesByResponse.delete(candidate);
      this.underrunFramesByResponse.delete(candidate);
      this.nextProgressFrameByResponse.delete(candidate);
      this.drainingResponseIds.delete(candidate);
      this.terminalFadeFramesByResponse.delete(candidate);
      if (this.activeResponseId === candidate) this.activeResponseId = null;
    });
    if (this.queue.length === 0) {
      this.started = false;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.rebuffering = false;
      this.fadeInFrames = 0;
    }
  }

  reportUnderrun(responseId) {
    const underrunFrames = this.underrunFramesByResponse.get(responseId) || 0;
    this.port.postMessage({
      type: 'playback-underrun',
      responseId,
      underrunMs: Math.round((underrunFrames * 1000) / sampleRate),
    });
  }

  recordPlayed(responseId, count) {
    const playedFrames = (this.playedFramesByResponse.get(responseId) || 0) + count;
    this.playedFramesByResponse.set(responseId, playedFrames);
    let nextProgressFrame = this.nextProgressFrameByResponse.get(responseId) || this.progressIntervalFrames;
    if (playedFrames < nextProgressFrame) return;
    this.port.postMessage({
      type: 'playback-progress',
      responseId,
      playedMs: Math.round((playedFrames * 1000) / sampleRate),
    });
    while (nextProgressFrame <= playedFrames) nextProgressFrame += this.progressIntervalFrames;
    this.nextProgressFrameByResponse.set(responseId, nextProgressFrame);
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.started) {
      if (
        this.rebuffering
        && this.activeResponseId
        && !this.drainingResponseIds.has(this.activeResponseId)
      ) {
        const underrunFrames = (this.underrunFramesByResponse.get(this.activeResponseId) || 0) + output.length;
        this.underrunFramesByResponse.set(this.activeResponseId, underrunFrames);
        this.reportUnderrun(this.activeResponseId);
      }
      if (this.queue.length > 0 && this.bufferWaitFrames > 0) {
        this.bufferWaitFrames = Math.max(0, this.bufferWaitFrames - output.length);
        return true;
      }
      if (this.queue.length > 0) {
        this.startPlayback();
        this.rebuffering = false;
      }
    }
    if (!this.started) {
      this.notifyIfDrained();
      return true;
    }

    let target = 0;
    while (target < output.length && this.queue.length > 0) {
      const entry = this.queue[0];
      const responseId = entry.responseId;
      if (this.activeResponseId !== responseId) this.activateResponse(responseId);
      const pcm = entry.pcm;
      const count = Math.min(output.length - target, pcm.length - this.offset);
      const remainingBeforeChunk = this.bufferedFrames(responseId);
      const terminalFadeFrames = this.terminalFadeFramesByResponse.get(responseId) || 0;
      for (let index = 0; index < count; index += 1) {
        let sample = pcm[this.offset + index] / 32768;
        if (this.fadeInFrames > 0) {
          const elapsed = this.fadeFrames - this.fadeInFrames;
          sample *= elapsed / this.fadeFrames;
          this.fadeInFrames -= 1;
        }
        const remainingFrames = remainingBeforeChunk - index;
        if (
          this.drainingResponseIds.has(responseId)
          && terminalFadeFrames > 0
          && remainingFrames <= terminalFadeFrames
        ) {
          sample *= terminalFadeFrames === 1
            ? 0
            : (remainingFrames - 1) / (terminalFadeFrames - 1);
        }
        output[target + index] = sample;
      }
      target += count;
      this.offset += count;
      this.recordPlayed(responseId, count);
      if (this.offset >= pcm.length) {
        this.queue.shift();
        this.offset = 0;
        this.notifyIfDrained(responseId);
      }
    }

    const responseId = this.activeResponseId;
    if (target < output.length && responseId && !this.drainingResponseIds.has(responseId)) {
      const fadeCount = Math.min(target, this.fadeFrames);
      for (let index = 0; index < fadeCount; index += 1) {
        output[target - fadeCount + index] *= (fadeCount - index - 1) / fadeCount;
      }
      const underrunFrames = (this.underrunFramesByResponse.get(responseId) || 0) + output.length - target;
      this.underrunFramesByResponse.set(responseId, underrunFrames);
      this.reportUnderrun(responseId);
      this.started = false;
      this.rebuffering = true;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.fadeInFrames = 0;
    }
    this.notifyIfDrained();
    return true;
  }
}

registerProcessor('fullduplex-pcm-playback', FullDuplexPcmPlayback);
