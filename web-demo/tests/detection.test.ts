import { expect, test } from "bun:test";
import { extract } from "../src/detection";

test("separates masks, excludes padding, restores coordinates and blends confidence", () => {
  const width = 30;
  const height = 20;

  const data = new Float32Array(width * height * 3).fill(-10);

  for (const [left, top, w, h, conf] of [
    [3, 3, 8, 8, 2],
    [17, 3, 8, 8, -2],
    [0, 0, 30, 1, 10],
  ]) {
    for (let y = top; y < top + h; y++) {
      for (let x = left; x < left + w; x++) {
        data[(y * width + x) * 3] = 2;
        data[(y * width + x) * 3 + 2] = conf;
      }
    }
  }

  const result = extract(
    data,
    width,
    height,
    { x: 2, y: 2, width: 26, height: 16 },
    { width: 52, height: 32 },
  );

  expect(result).toHaveLength(2);
  expect(result[0].box).toEqual([2, 2, 16, 16]);
  expect(result[0].score).toBeCloseTo(0.8808, 3);
  expect(result[1].score).toBeCloseTo(0.5);
  expect(result[0].path).toContain("M2.0,2.0");
  expect(result[0].path.endsWith("Z")).toBe(true);
});

test("empty masks and isolated noise yield no bubbles", () => {
  const data = new Float32Array(300).fill(-10);
  data[0] = 10;

  const result = extract(
    data,
    10,
    10,
    { x: 0, y: 0, width: 10, height: 10 },
    { width: 10, height: 10 },
  );

  expect(result).toEqual([]);
});
