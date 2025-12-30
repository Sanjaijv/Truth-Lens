import "./modulepreload-polyfill.js";
console.log("TruthLens: Frame Loaded. Origin:", window.location.origin);
let worker = null;
let parentSource = null;
let parentOrigin = null;
window.addEventListener("message", async (event) => {
  const { type, data } = event.data;
  if (type === "INIT_WORKER") {
    parentSource = event.source;
    parentOrigin = event.origin;
    console.log("TruthLens: Frame received INIT from", parentOrigin);
    try {
      const workerUrl = chrome.runtime.getURL("dist/analysis.worker.js");
      console.log("TruthLens: Spawning Worker:", workerUrl);
      worker = new Worker(workerUrl);
      worker.onmessage = (wEvent) => {
        console.log("TruthLens Frame: Msg from Worker", wEvent.data);
        if (parentSource) {
          parentSource.postMessage({
            source: "truthlens-worker",
            ...wEvent.data
          }, parentOrigin);
        }
      };
      worker.onerror = (err) => {
        const msg = err.message || JSON.stringify(err);
        console.error("TruthLens Worker Error in Frame:", err);
        if (parentSource) {
          parentSource.postMessage({
            source: "truthlens-worker",
            type: "ERROR",
            data: { error: msg }
          }, parentOrigin);
        }
      };
      worker.postMessage({ type: "INIT", rootUrl: chrome.runtime.getURL("") });
      parentSource.postMessage({ source: "truthlens-worker", type: "frame-ready" }, parentOrigin);
    } catch (e) {
      console.error("TruthLens: Failed to spawn worker in frame:", e);
      if (parentSource) {
        parentSource.postMessage({
          source: "truthlens-worker",
          type: "ERROR",
          data: { error: "Frame failed to spawn worker: " + e.message }
        }, parentOrigin);
      }
    }
  } else if (worker) {
    if (event.data.pixels) {
      worker.postMessage(event.data, [event.data.pixels]);
    } else {
      worker.postMessage(event.data);
    }
  }
});
