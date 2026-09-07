export interface Bubble {
  score: number;
  path: string;
  box: [number, number, number, number];
}

export type Backend = "webgpu" | "wasm";

export interface Result {
  bubbles: Bubble[];
  milliseconds: number;
  width: number;
  height: number;
  backend?: Backend;
}

export interface Job {
  id: number;
  bitmap: ImageBitmap;
  backend?: Backend;
}

export interface Reply {
  id: number;
  result?: Result;
  error?: string;
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * Extract connected masks and trace their pixel-edge contours in image coordinates.
 * Unlike the Python reference, this lightweight browser implementation does not
 * use OpenCV watershed; touching masks may remain a single bubble.
 */
export function extract(
  logits: Float32Array,
  width: number,
  height: number,
  crop: { x: number; y: number; width: number; height: number },
  original: { width: number; height: number },
): Bubble[] {
  const labels = new Int32Array(width * height);
  const queue = new Int32Array(width * height);
  let label = 0;
  const bubbles: Bubble[] = [];

  const sx = original.width / crop.width;
  const sy = original.height / crop.height;

  for (let y = crop.y; y < crop.y + crop.height; y++) {
    for (let x = crop.x; x < crop.x + crop.width; x++) {
      const start = y * width + x;

      if (labels[start] || logits[start * 3] < 0) {
        continue;
      }

      label++;

      let head = 0;
      let tail = 1;
      let score = 0;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;

      queue[0] = start;
      labels[start] = label;

      while (head < tail) {
        const p = queue[head++];
        const px = p % width;
        const py = Math.floor(p / width);

        score += (sigmoid(logits[p * 3]) + sigmoid(logits[p * 3 + 2])) / 2;

        minX = Math.min(minX, px);
        maxX = Math.max(maxX, px);
        minY = Math.min(minY, py);
        maxY = Math.max(maxY, py);

        for (const [dx, dy] of [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ]) {
          const nx = px + dx;
          const ny = py + dy;
          const next = ny * width + nx;

          if (
            nx < crop.x ||
            nx >= crop.x + crop.width ||
            ny < crop.y ||
            ny >= crop.y + crop.height
          ) {
            continue;
          }

          if (!labels[next] && logits[next * 3] >= 0) {
            labels[next] = label;
            queue[tail++] = next;
          }
        }
      }

      if (tail * sx * sy < 60) {
        continue;
      }

      // Directed perimeter edges, linked into closed loops (including holes).
      const edges = new Map<number, number[]>();
      const stride = width + 1;

      const edge = (ax: number, ay: number, bx: number, by: number) => {
        const a = ay * stride + ax;
        const b = by * stride + bx;
        const list = edges.get(a) ?? [];
        list.push(b);
        edges.set(a, list);
      };

      for (let i = 0; i < tail; i++) {
        const p = queue[i];
        const px = p % width;
        const py = Math.floor(p / width);

        if (py === 0 || labels[p - width] !== label) {
          edge(px, py, px + 1, py);
        }

        if (px === width - 1 || labels[p + 1] !== label) {
          edge(px + 1, py, px + 1, py + 1);
        }

        if (py === height - 1 || labels[p + width] !== label) {
          edge(px + 1, py + 1, px, py + 1);
        }

        if (px === 0 || labels[p - 1] !== label) {
          edge(px, py + 1, px, py);
        }
      }

      let path = "";

      while (edges.size) {
        const first = edges.keys().next().value!;
        let p = first;
        const points: number[] = [];

        do {
          points.push(p);
          const next = edges.get(p);

          if (!next) {
            break;
          }

          const from = p;
          p = next.pop()!;

          if (!next.length) {
            edges.delete(from);
          }
        } while (p !== first);

        const reduced = points.filter((point, i) => {
          const prev = points[(i + points.length - 1) % points.length];
          const next = points[(i + 1) % points.length];
          return point - prev !== next - point;
        });

        path +=
          reduced
            .map(
              (point, i) =>
                `${i ? "L" : "M"}${(((point % stride) - crop.x) * sx).toFixed(1)},${((Math.floor(point / stride) - crop.y) * sy).toFixed(1)}`,
            )
            .join("") + "Z";
      }

      bubbles.push({
        score: score / tail,
        path,
        box: [
          (minX - crop.x) * sx,
          (minY - crop.y) * sy,
          (maxX + 1 - minX) * sx,
          (maxY + 1 - minY) * sy,
        ],
      });
    }
  }

  return bubbles.sort((a, b) => b.score - a.score);
}
