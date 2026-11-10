import { decodeModel, restorePlane } from "./lut.mjs";

self.onmessage = async ({ data: { rgba, width, height, task, stacks } }) => {
  try {
    const sr = task === "sr";
    const scale = sr ? 4 : 1;
    const channels = sr ? 3 : 1;
    const filename = sr ? `sr-${stacks}` : task;
    const response = await fetch(new URL(`./models/${filename}.bin`, import.meta.url));
    if (!response.ok) throw new Error("Could not load the model. Please try again.");
    const model = decodeModel(await response.arrayBuffer(), sr ? stacks : 7, scale);
    const outWidth = width * scale;
    const outHeight = height * scale;
    const restored = new Uint8ClampedArray(outWidth * outHeight * 4);
    let passes = 0;
    for (let channel = 0; channel < channels; channel++) {
      const plane = Uint8Array.from({ length: width * height }, (_, i) => rgba[i * 4 + channel]);
      const result = restorePlane(plane, width, height, model, () => {
        passes += 1;
        self.postMessage({ progress: passes / (4 * channels) });
      });
      for (let i = 0; i < result.length; i++) {
        if (sr) restored[i * 4 + channel] = result[i];
        else restored.fill(result[i], i * 4, i * 4 + 3);
        restored[i * 4 + 3] = 255;
      }
    }
    self.postMessage({ restored, width: outWidth, height: outHeight }, [restored.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
