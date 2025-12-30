// frame.js - Proxies messages between Content Script and background Worker
// Runs in extension origin, so it CAN spawn workers safely.

console.log("TruthLens: Frame Loaded. Origin:", window.location.origin);

let worker = null;
let parentSource = null;
let parentOrigin = null;

// 1. Listen for messages from Content Script (Parent)
window.addEventListener('message', async (event) => {
    // Basic security check: ensure message is from a trusted context (though iframe is usually isolated)
    // For MV3 content script, event.origin will be the page origin (e.g. https://www.youtube.com)

    const { type, data } = event.data;

    // Handshake / Init
    if (type === 'INIT_WORKER') {
        parentSource = event.source;
        parentOrigin = event.origin;
        console.log("TruthLens: Frame received INIT from", parentOrigin);

        try {
            // Spawn the actual Worker (relative path in dist/)
            // We assume frame.html and analysis.worker.js are in the same 'dist' folder
            // or we use the full chrome-extension URL.
            const workerUrl = chrome.runtime.getURL('dist/analysis.worker.js');
            console.log("TruthLens: Spawning Worker:", workerUrl);

            worker = new Worker(workerUrl);

            // Forward messages from Worker back to Parent
            worker.onmessage = (wEvent) => {
                console.log("TruthLens Frame: Msg from Worker", wEvent.data);
                if (parentSource) {
                    parentSource.postMessage({
                        source: 'truthlens-worker',
                        ...wEvent.data
                    }, parentOrigin);
                }
            };

            worker.onerror = (err) => {
                const msg = err.message || JSON.stringify(err);
                console.error("TruthLens Worker Error in Frame:", err);
                if (parentSource) {
                    parentSource.postMessage({
                        source: 'truthlens-worker',
                        type: 'ERROR',
                        data: { error: msg }
                    }, parentOrigin);
                }
            };

            // Pass initialization down to worker
            worker.postMessage({ type: 'INIT', rootUrl: chrome.runtime.getURL('') });

            parentSource.postMessage({ source: 'truthlens-worker', type: 'frame-ready' }, parentOrigin);

        } catch (e) {
            console.error("TruthLens: Failed to spawn worker in frame:", e);
            if (parentSource) {
                parentSource.postMessage({
                    source: 'truthlens-worker',
                    type: 'ERROR',
                    data: { error: "Frame failed to spawn worker: " + e.message }
                }, parentOrigin);
            }
        }
    }

    // Forward generic messages to Worker
    else if (worker) {
        // Forward buffer if present
        if (event.data.pixels) {
            worker.postMessage(event.data, [event.data.pixels]);
        } else {
            worker.postMessage(event.data);
        }
    }
});
