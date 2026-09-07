import * as ort from "onnxruntime-web/webgpu";
import { extract, type Backend, type Job, type Reply } from "./detection";

const base = new URL(
  import.meta.env.BASE_URL,
  self.location.href.replace(/\/(assets|src)\/[^/]*$/, "/"),
);
ort.env.wasm.wasmPaths = new URL("ort/", base).href;
ort.env.wasm.numThreads = 1;

const sessions = new Map<Backend, Promise<ort.InferenceSession>>();

async function getSession(backend: Backend): Promise<ort.InferenceSession> {
  const existing = sessions.get(backend);

  if (existing) {
    return existing;
  }

  const promise = ort.InferenceSession.create(
    new URL("assets/model.onnx", base).href,
    { executionProviders: [backend] },
  );

  sessions.set(backend, promise);

  try {
    return await promise;
  } catch (error) {
    sessions.delete(backend);
    throw error;
  }
}

self.onmessage = async ({
  data: { id, bitmap, backend = "webgpu" },
}: MessageEvent<Job>) => {
  try {
    let activeBackend: Backend = backend;
    let model: ort.InferenceSession;

    if (backend === "webgpu") {
      try {
        model = await getSession("webgpu");
      } catch (err) {
        console.warn(
          "WebGPU session creation failed, falling back to WASM:",
          err,
        );
        model = await getSession("wasm");
        activeBackend = "wasm";
      }
    } else {
      model = await getSession("wasm");
      activeBackend = "wasm";
    }

    const start = performance.now();
    const width = 768;
    const height = 1024;

    const scale = Math.min(width / bitmap.width, height / bitmap.height);
    const rw = Math.max(1, Math.round(bitmap.width * scale));
    const rh = Math.max(1, Math.round(bitmap.height * scale));

    const x = Math.floor((width - rw) / 2);
    const y = Math.floor((height - rh) / 2);

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(bitmap, x, y, rw, rh);

    const rgba = ctx.getImageData(0, 0, width, height).data;
    const input = new Float32Array(width * height);

    for (let i = 0; i < input.length; i++) {
      input[i] =
        ((0.299 * rgba[i * 4] +
          0.587 * rgba[i * 4 + 1] +
          0.114 * rgba[i * 4 + 2]) /
          255 -
          0.449) /
        0.226;
    }

    const tensor = new ort.Tensor("float32", input, [1, height, width, 1]);
    let outputs: Record<string, ort.Tensor> | undefined;

    try {
      if (activeBackend === "webgpu") {
        try {
          outputs = await model.run({ [model.inputNames[0]]: tensor });
        } catch (runErr) {
          console.warn(
            "WebGPU inference execution failed, falling back to WASM:",
            runErr,
          );
          sessions.delete("webgpu");
          model = await getSession("wasm");
          activeBackend = "wasm";
          outputs = await model.run({ [model.inputNames[0]]: tensor });
        }
      } else {
        outputs = await model.run({ [model.inputNames[0]]: tensor });
      }

      const logits = outputs[model.outputNames[0]];

      if (logits.dims.join(",") !== `1,${height},${width},3`) {
        throw new Error("Unexpected model output dimensions");
      }

      const bubbles = extract(
        logits.data as Float32Array,
        width,
        height,
        { x, y, width: rw, height: rh },
        bitmap,
      );

      self.postMessage({
        id,
        result: {
          bubbles,
          width: bitmap.width,
          height: bitmap.height,
          milliseconds: performance.now() - start,
          backend: activeBackend,
        },
      } satisfies Reply);
    } finally {
      tensor.dispose();

      if (outputs) {
        Object.values(outputs).forEach((t) => {
          t.dispose();
        });
      }
    }
  } catch (error) {
    sessions.clear();

    self.postMessage({
      id,
      error: `Detection failed: ${error instanceof Error ? error.message : String(error)}. Drop or select an image to retry.`,
    } satisfies Reply);
  } finally {
    bitmap.close();
  }
};
