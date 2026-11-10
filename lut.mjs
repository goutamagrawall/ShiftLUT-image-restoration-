// Each table entry is the original interpolated value multiplied by 64.
const CHANNELS = 16;
const WEIGHT_SCALE = 64;
const clamp = (value, low, high) => Math.max(low, Math.min(value, high));

export function decodeModel(buffer, stacks, scale) {
  const words = new Int16Array(buffer);
  let cursor = 0;
  function read(count) {
    if (cursor + count > words.length) throw new Error("Incomplete model tables.");
    const values = words.subarray(cursor, cursor + count);
    cursor += count;
    return values;
  }
  const offsets = read((stacks + 1) * 32);
  const lsb = read(9 * 4 * CHANNELS);
  const layers = Array.from({ length: stacks + 1 }, () => ({
    depth: read(9 * 64 * CHANNELS),
    point: read(CHANNELS * 64 * CHANNELS),
  }));
  const up = read(CHANNELS * 64 * scale * scale);
  if (cursor !== words.length) throw new Error("Unexpected model table size.");
  return { offsets, lsb, layers, up, scale };
}

function roundEven(numerator, denominator) {
  const whole = Math.floor(numerator / denominator);
  const remainder = 2 * (numerator - whole * denominator);
  return whole + Number(remainder > denominator || (remainder === denominator && (whole & 1)));
}

function rotate(input, width, height, turns) {
  if (turns === 0) return { pixels: input, width, height };
  const pixels = new input.constructor(input.length);
  const outWidth = turns % 2 ? height : width;
  const outHeight = turns % 2 ? width : height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = turns === 1 ? y : turns === 2 ? width - 1 - x : height - 1 - y;
      const dy = turns === 1 ? width - 1 - x : turns === 2 ? height - 1 - y : x;
      pixels[dy * outWidth + dx] = input[y * width + x];
    }
  }
  return { pixels, width: outWidth, height: outHeight };
}

function neighborsFor(width, height) {
  const indices = new Int32Array(width * height * 9);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let index = (y * width + x) * 9;
      for (let dy = -1; dy <= 1; dy++) {
        const row = clamp(y + dy, 0, height - 1) * width;
        for (let dx = -1; dx <= 1; dx++) indices[index++] = row + clamp(x + dx, 0, width - 1);
      }
    }
  }
  return indices;
}

function depthwise(table, values, neighbors, pixel, channel, stride, domain, bias) {
  let sum = 0;
  for (let tap = 0; tap < 9; tap++) {
    const index = neighbors[pixel * 9 + tap] * stride + (stride === 1 ? 0 : channel);
    sum += table[(tap * domain + values[index] + bias) * CHANNELS + channel];
  }
  return roundEven(sum, 9 * WEIGHT_SCALE);
}

function pointwise(table, values, pixel, outputs, sums) {
  sums.fill(0);
  for (let channel = 0; channel < CHANNELS; channel++) {
    const base = (channel * 64 + values[pixel * CHANNELS + channel] + 32) * outputs;
    for (let out = 0; out < outputs; out++) sums[out] += table[base + out];
  }
}

function forward(pixels, width, height, model) {
  const { offsets, lsb, layers, up, scale } = model;
  const count = width * height;
  const neighbors = neighborsFor(width, height);
  const low = new Int8Array(count);
  const firstHigh = new Int8Array(count);
  for (let p = 0; p < count; p++) {
    low[p] = pixels[p] & 3;
    firstHigh[p] = (pixels[p] >> 2) - 32;
  }
  let high = firstHigh;
  const depth = new Int8Array(count * CHANNELS);
  const shifted = new Int8Array(depth.length);
  const features = new Int8Array(depth.length);
  const sums = new Int32Array(CHANNELS);
  for (let layer = 0; layer < layers.length; layer++) {
    const stride = layer === 0 ? 1 : CHANNELS;
    for (let p = 0; p < count; p++) {
      for (let c = 0; c < CHANNELS; c++) {
        const index = p * CHANNELS + c;
        const residual = stride === 1 ? firstHigh[p] : high[index];
        const fine = layer === 0
          ? clamp(low[p] + depthwise(lsb, low, neighbors, p, c, 1, 4, 0), 0, 3)
          : 0;
        depth[index] = clamp(residual + fine + depthwise(layers[layer].depth, high, neighbors, p, c, stride, 64, 32), -32, 31);
      }
    }
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < CHANNELS; c++) {
          const row = clamp(y + offsets[layer * 32 + c], 0, height - 1);
          const col = clamp(x + offsets[layer * 32 + CHANNELS + c], 0, width - 1);
          shifted[(y * width + x) * CHANNELS + c] = depth[(row * width + col) * CHANNELS + c];
        }
      }
    }
    for (let p = 0; p < count; p++) {
      pointwise(layers[layer].point, shifted, p, CHANNELS, sums);
      for (let c = 0; c < CHANNELS; c++) {
        const index = p * CHANNELS + c;
        features[index] = clamp(shifted[index] + roundEven(sums[c], CHANNELS * WEIGHT_SCALE), -32, 31);
      }
    }
    high = features;
  }
  const output = new Int16Array(count * scale * scale);
  const outputs = scale * scale;
  const finalSums = new Int32Array(outputs);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      pointwise(up, high, pixel, outputs, finalSums);
      for (let c = 0; c < outputs; c++) {
        const row = y * scale + Math.floor(c / scale);
        const col = x * scale + c % scale;
        const residual = scale === 4 ? high[pixel * CHANNELS + c] : 0;
        output[row * width * scale + col] = clamp(residual + roundEven(finalSums[c], CHANNELS * WEIGHT_SCALE), -128, 127);
      }
    }
  }
  return output;
}

export function restorePlane(input, width, height, model, onPass = () => {}) {
  const { scale } = model;
  const sum = new Int16Array(input.length * scale * scale);
  for (let turn = 0; turn < 4; turn++) {
    const rotated = rotate(input, width, height, turn);
    const output = forward(rotated.pixels, rotated.width, rotated.height, model);
    const aligned = rotate(output, rotated.width * scale, rotated.height * scale, (4 - turn) % 4).pixels;
    for (let i = 0; i < sum.length; i++) sum[i] += aligned[i];
    onPass();
  }
  return Uint8ClampedArray.from(sum, (value, i) => value + (scale === 4 ? 128 : input[i]));
}
