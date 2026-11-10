const el = (id) => document.getElementById(id);
const before = el("before");
const after = el("after");
const download = el("download");
const restore = el("restore");
const progress = el("progress");
const status = el("status");
const sampleButtons = [...el("sample-gallery").querySelectorAll("button")];
const maxPixels = 1_048_576;
const explanations = {
  sr: "Give a small image more pixels while preserving color.",
  denoising: "Reduce Gaussian noise. The supplied model produces a grayscale image.",
  deblocking: "Reduce strong JPEG compression artifacts. The supplied model produces a grayscale image.",
};
const task = () => document.querySelector('input[name="task"]:checked').value;
const context = (canvas) => canvas.getContext("2d", { willReadFrequently: true });
let source = null;
let activeSample = null;
let fileName = "";
let worker = null;
let generation = 0;
let downloadUrl = null;

function showError(message = "") {
  el("error").textContent = message;
  el("error").hidden = !message;
}

function clearResult() {
  if (worker) worker.terminate();
  worker = null;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
  download.removeAttribute("href");
  download.hidden = true;
  el("after-card").hidden = true;
  el("views").hidden = true;
  el("wipe-area").hidden = true;
  el("view").value = "side";
  after.width = after.height = 1;
  el("wipe").width = el("wipe").height = 1;
  progress.hidden = true;
  restore.textContent = "Restore image";
  status.textContent = "";
  showError();
}

function prepareInput() {
  clearResult();
  const selected = task();
  el("description").textContent = explanations[selected];
  el("variant-setting").hidden = selected !== "sr";
  el("filename").textContent = source ? fileName : "";
  el("dimensions").textContent = "";
  el("empty").hidden = Boolean(source);
  el("side").hidden = !source;
  restore.disabled = true;
  if (!source) {
    before.width = before.height = 1;
    return;
  }
  const maxEdge = Number(el("size").value);
  const ratio = maxEdge ? Math.min(1, maxEdge / Math.max(source.width, source.height)) : 1;
  const width = Math.max(1, Math.round(source.width * ratio));
  const height = Math.max(1, Math.round(source.height * ratio));
  if (width * height > maxPixels) {
    before.width = before.height = 1;
    el("side").hidden = true;
    el("empty").hidden = false;
    showError("Choose a smaller processing size for this image. Original size supports up to 1 megapixel.");
    return;
  }
  before.width = width;
  before.height = height;
  const ctx = context(before);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  if (selected !== "sr") {
    const frame = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < frame.data.length; i += 4) {
      const gray = Math.round((299 * frame.data[i] + 587 * frame.data[i + 1] + 114 * frame.data[i + 2]) / 1000);
      frame.data.fill(gray, i, i + 3);
    }
    ctx.putImageData(frame, 0, 0);
  }
  const scale = selected === "sr" ? 4 : 1;
  const prefix = ratio < 1 ? `Resized from ${source.width} × ${source.height}. ` : "";
  el("dimensions").textContent = `${prefix}Processing ${width} × ${height}. Output ${width * scale} × ${height * scale}.`;
  restore.disabled = false;
}

async function openSource(load, name, sample = null) {
  const request = ++generation;
  source?.close?.();
  source = null;
  activeSample = sample;
  for (const button of sampleButtons) {
    button.setAttribute("aria-pressed", String(button === activeSample));
  }
  prepareInput();
  status.textContent = "Opening image…";
  try {
    const image = await load();
    if (generation !== request) {
      image.close?.();
      return;
    }
    source = image;
    fileName = name;
    prepareInput();
  } catch (error) {
    if (generation !== request) return;
    status.textContent = "";
    showError(error.message);
  }
}

function useFile(file) {
  if (!file) return;
  openSource(async () => {
    if (file.size > 20 * 1024 * 1024) throw new Error("Choose a file up to 20 MB.");
    let image;
    try {
      image = await createImageBitmap(file);
    } catch {
      throw new Error("This image could not be opened. Try a PNG, JPEG, WebP, or BMP file.");
    }
    if (image.width * image.height > 40_000_000) {
      image.close();
      throw new Error("Choose an image up to 40 megapixels.");
    }
    return image;
  }, file.name);
}

function useSample(button) {
  const selected = task();
  const thumbnail = button.querySelector("img");
  const label = button.textContent.trim();
  openSource(async () => {
    try {
      await thumbnail.decode();
    } catch {
      throw new Error("Could not load this sample. Refresh the page or choose another image.");
    }
    const width = thumbnail.naturalWidth;
    const height = thumbnail.naturalHeight;
    if (selected === "sr") {
      return createImageBitmap(thumbnail, {
        resizeWidth: Math.floor(width / 4),
        resizeHeight: Math.floor(height / 4),
        resizeQuality: "high",
      });
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = context(canvas);
    ctx.drawImage(thumbnail, 0, 0);
    if (selected === "deblocking") {
      const jpeg = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.1));
      return createImageBitmap(jpeg);
    }
    const frame = ctx.getImageData(0, 0, width, height);
    let seed = 23;
    const random = () => {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      return (seed + 1) / 4294967297;
    };
    for (let i = 0; i < frame.data.length; i += 4) {
      const noise = 15 * Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
      for (let c = 0; c < 3; c++) frame.data[i + c] += noise;
    }
    ctx.putImageData(frame, 0, 0);
    return createImageBitmap(canvas);
  }, `${label} sample`, button);
}

function updateComparison() {
  const wipeMode = el("view").value === "wipe";
  el("side").hidden = wipeMode;
  el("wipe-area").hidden = !wipeMode;
  if (!wipeMode) return;
  const canvas = el("wipe");
  const ratio = Math.min(1, 900 / after.width, 460 / after.height);
  canvas.width = Math.max(1, Math.round(after.width * ratio));
  canvas.height = Math.max(1, Math.round(after.height * ratio));
  const ctx = context(canvas);
  const cut = Math.round(canvas.width * Number(el("split").value) / 100);
  ctx.drawImage(after, 0, 0, canvas.width, canvas.height);
  if (cut > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, cut, canvas.height);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(before, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }
  if (cut > 0 && cut < canvas.width) {
    ctx.fillStyle = "#087469";
    ctx.fillRect(cut - 1, 0, 2, canvas.height);
  }
}

restore.addEventListener("click", () => {
  clearResult();
  const selected = task();
  const { width, height } = before;
  const rgba = context(before).getImageData(0, 0, width, height).data;
  const start = performance.now();
  const job = new Worker(new URL("./worker.mjs", import.meta.url), { type: "module" });
  worker = job;
  restore.disabled = true;
  restore.textContent = "Restoring…";
  status.textContent = "Loading the model and restoring your image…";
  progress.value = 0;
  progress.hidden = false;

  function fail(message) {
    if (worker !== job) return;
    clearResult();
    restore.disabled = false;
    showError(message);
  }

  job.onerror = () => fail("Restoration stopped. Try a smaller processing size.");
  job.onmessage = ({ data }) => {
    if (worker !== job) return;
    if (data.error) return fail(data.error);
    if (data.progress !== undefined) {
      progress.value = data.progress;
      status.textContent = `Restoring your image… ${Math.round(data.progress * 100)}%`;
      return;
    }
    after.width = data.width;
    after.height = data.height;
    context(after).putImageData(new ImageData(data.restored, data.width, data.height), 0, 0);
    after.toBlob((png) => {
      if (worker !== job) return;
      if (!png) return fail("Could not make a PNG from the result. Try a smaller processing size.");
      job.terminate();
      worker = null;
      downloadUrl = URL.createObjectURL(png);
      download.href = downloadUrl;
      const stem = fileName.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 100) || "image";
      download.download = `${stem}-${selected}.png`;
      download.hidden = false;
      el("after-card").hidden = false;
      el("views").hidden = false;
      progress.hidden = true;
      restore.disabled = false;
      restore.textContent = "Restore image";
      status.textContent = `Restored ${data.width} × ${data.height} in ${((performance.now() - start) / 1000).toFixed(1)} seconds.`;
    }, "image/png");
  };
  job.postMessage({ rgba, width, height, task: selected, stacks: Number(el("variant").value) }, [rgba.buffer]);
});

el("upload").addEventListener("change", (event) => {
  useFile(event.target.files[0]);
  event.target.value = "";
});
for (const button of sampleButtons) {
  button.addEventListener("click", () => useSample(button));
}
el("tasks").addEventListener("change", () => activeSample ? useSample(activeSample) : prepareInput());
el("size").addEventListener("change", prepareInput);
el("variant").addEventListener("change", prepareInput);
el("view").addEventListener("change", updateComparison);
el("split").addEventListener("input", updateComparison);
el("drop").addEventListener("dragover", (event) => {
  event.preventDefault();
  el("drop").classList.add("dragging");
});
el("drop").addEventListener("dragleave", () => el("drop").classList.remove("dragging"));
el("drop").addEventListener("drop", (event) => {
  event.preventDefault();
  el("drop").classList.remove("dragging");
  useFile(event.dataTransfer.files[0]);
});
prepareInput();
