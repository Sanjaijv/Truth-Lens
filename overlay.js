// overlay.js - Full Screen Analysis Modal

export class TruthLensOverlay {
    constructor() {
        this.element = null;
        this.mirrorVideo = null;
        this.canvas = null;
        this.onClose = null;
    }

    createOverlay(targetParent) { // targetParent ignored, strictly body
        if (this.element) this.remove();

        // Container
        this.element = document.createElement('div');
        this.element.className = 'truthlens-modal-overlay';

        // HTML Structure
        this.element.innerHTML = `
            <div class="truthlens-modal-header">
                <span class="truthlens-modal-title">TruthLens</span>
                <button class="truthlens-modal-close-btn" id="tl-modal-close">Exit</button>
            </div>

            <div class="truthlens-video-wrapper">
                <canvas class="truthlens-overlay-canvas" style="pointer-events: none;"></canvas>
            </div>

            <div class="truthlens-modal-footer">
                <div class="truthlens-modal-status analyzing" id="tl-modal-status">
                    <span class="truthlens-indicator"></span>
                    <span id="tl-status-text">Initializing...</span>
                </div>
            </div>
            
            <div id="tl-debug-console" class="truthlens-debug-console"></div>
        `;

        document.body.appendChild(this.element);

        // References
        // this.mirrorVideo removed - using canvas drawImage
        this.canvas = this.element.querySelector('.truthlens-overlay-canvas');

        // Binds
        this.element.querySelector('#tl-modal-close').onclick = () => {
            if (this.onClose) this.onClose();
            this.remove();
        };
    }

    updateStatus(text, statusClass = 'analyzing') {
        if (!this.element) return;
        const statusEl = this.element.querySelector('#tl-modal-status');
        const textEl = this.element.querySelector('#tl-status-text');

        // Remove old classes
        statusEl.classList.remove('analyzing', 'safe', 'fake', 'suspicious');
        statusEl.classList.add(statusClass);
        textEl.textContent = text;
    }

    log(msg) {
        if (!this.element) return;
        const consoleEl = this.element.querySelector('#tl-debug-console');
        if (consoleEl) {
            const line = document.createElement('div');

            // Highlight keywords
            let formattedMsg = msg
                .replace(/(CORS|tainting|SecurityError|Protected Content)/gi, '<span class="tl-log-cors">$1</span>')
                .replace(/(Error|Failed|Exception)/gi, '<span class="tl-log-error">$1</span>')
                .replace(/(Warning|Suspicious)/gi, '<span class="tl-log-warn">$1</span>')
                .replace(/(Info|Loaded|Ready)/gi, '<span class="tl-log-info">$1</span>');

            line.innerHTML = `> ${formattedMsg}`;
            consoleEl.appendChild(line);
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
    }

    showVerdict(label, score, onRescan) {
        if (!this.element) return;

        let statusClass = 'safe';
        if (label === 'Fake') statusClass = 'fake';
        else if (label === 'Suspicious') statusClass = 'suspicious';

        this.updateStatus(`Analysis Complete: ${label} (${score}% Confidence)`, statusClass);

        // Maybe change the Close button to "Done" or add Rescan?
        // For simplicity, just updating text is usually enough for the modal model.
        // User can click Exit to leave.
    }

    showError(message) {
        if (!this.element) return;

        const footer = this.element.querySelector('.truthlens-modal-footer');
        if (!footer) return;

        let title = "Analysis Error";
        let desc = message;
        // Detect CORS/Protected content specifically
        const isCORS = message.includes("CORS") || message.includes("Protected") || message.includes("SecurityError") || message.includes("Tainted");

        if (isCORS) {
            title = "Protected Content (CORS)";
            desc = "This video is protected by browser security policies and cannot be analyzed directly. Try opening the video in a new tab or use a different source.";
        }

        footer.innerHTML = `
            <div class="truthlens-error-alert">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="truthlens-error-icon"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <div class="truthlens-error-content">
                    <span class="truthlens-error-title">${title}</span>
                    <span class="truthlens-error-desc">${desc}</span>
                </div>
            </div>
        `;

        // Also log it
        this.log(`Error: ${message}`);
    }

    remove() {
        if (this.element) {
            this.element.remove();
            this.element = null;
            this.canvas = null;
        }
    }
}
