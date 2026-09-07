import { createReadStream, existsSync, statSync } from "node:fs";
import { cp, mkdir } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

const MIME_TYPES: Record<string, string> = {
  ".wasm": "application/wasm",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".onnx": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".json": "application/json",
  ".css": "text/css",
  ".svg": "image/svg+xml",
};

function staticAssetsPlugin(): Plugin {
  return {
    name: "static-assets-plugin",

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.method !== "GET" && req.method !== "HEAD") {
          return next();
        }

        const url = (req.url || "").split("?")[0];
        let filePath: string | null = null;

        if (url.startsWith("/assets/")) {
          filePath = resolve(import.meta.dirname, "." + url);
        } else if (url.startsWith("/ort/")) {
          const filename = url.replace(/^\/ort\//, "");
          filePath = resolve(
            import.meta.dirname,
            "node_modules/onnxruntime-web/dist",
            filename,
          );
        }

        if (
          filePath &&
          existsSync(filePath) &&
          !statSync(filePath).isDirectory()
        ) {
          const ext = extname(filePath);
          const contentType = MIME_TYPES[ext] || "application/octet-stream";
          res.setHeader("Content-Type", contentType);
          createReadStream(filePath).pipe(res);
          return;
        }

        next();
      });
    },

    async closeBundle() {
      const distOrt = resolve(import.meta.dirname, "dist/ort");
      const distAssets = resolve(import.meta.dirname, "dist/assets");

      await mkdir(distOrt, { recursive: true });
      await mkdir(distAssets, { recursive: true });

      await cp(resolve(import.meta.dirname, "assets"), distAssets, {
        recursive: true,
      });

      const ortFiles = [
        "ort-wasm-simd-threaded.jsep.wasm",
        "ort-wasm-simd-threaded.jsep.mjs",
        "ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd-threaded.mjs",
      ];

      for (const name of ortFiles) {
        await cp(
          resolve(
            import.meta.dirname,
            "node_modules/onnxruntime-web/dist",
            name,
          ),
          resolve(distOrt, name),
        );
      }
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [staticAssetsPlugin()],
  server: {
    port: 3000,
    host: "127.0.0.1",
  },
  preview: {
    port: 3000,
    host: "127.0.0.1",
  },
});
