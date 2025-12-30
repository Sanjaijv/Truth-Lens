// utils/frameSampler.js
export class FrameSampler {
    constructor(videoElement, canvas) {
        this.video = videoElement;
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    captureFrame() {
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        return this.canvas;
    }
}
