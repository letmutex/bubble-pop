import type { Backend, Bubble, Reply, Result } from "./detection";

const $ = <T extends HTMLElement>(
  selector: string,
  root: ParentNode = document,
) => root.querySelector<T>(selector)!;

const book = $("#book");
const options = $("#options");
const optionsButton = $("#options-button");

const threshold = $<HTMLInputElement>("#threshold");
const outline = $<HTMLInputElement>("#outline");
const confidence = $<HTMLInputElement>("#confidence");
const webgpu = $<HTMLInputElement>("#webgpu");
const webgpuLabel = $("#webgpu-label");
const webgpuRow = $<HTMLLabelElement>('label[for="webgpu"]');

async function checkWebGPUSupport(): Promise<boolean> {
  if (
    typeof navigator === "undefined" ||
    !("gpu" in navigator) ||
    !navigator.gpu
  ) {
    return false;
  }

  try {
    const gpu = navigator.gpu as {
      requestAdapter?: () => Promise<unknown>;
    };

    if (typeof gpu.requestAdapter !== "function") {
      return false;
    }

    const adapterPromise = gpu.requestAdapter();
    const timeoutPromise = new Promise<null>((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, 2000);
    });

    const adapter = await Promise.race([adapterPromise, timeoutPromise]);
    return adapter !== null;
  } catch {
    return false;
  }
}

const webgpuSupportedPromise = checkWebGPUSupport();

async function initWebGPU() {
  const supported = await webgpuSupportedPromise;

  if (!supported) {
    webgpu.checked = false;
    webgpu.disabled = true;
    webgpuRow.classList.add("disabled");
    webgpuRow.title = "WebGPU is not supported by your browser or device.";
    webgpuLabel.textContent = "WebGPU (unavailable)";
  }
}

void initWebGPU();

const popup = $<HTMLDivElement>("#bubble-popup");
const popupCanvas = $<HTMLCanvasElement>("#popup-canvas");
const mobile = matchMedia("(max-width: 640px)");

let activeBubble: Bubble | undefined;
let activePage: Page | undefined;
let hideTimer: ReturnType<typeof setTimeout> | undefined;
let isPopupVisible = false;

type Page = {
  root: HTMLElement;
  img: HTMLImageElement;
  status: HTMLElement;
  input: HTMLInputElement;
  revision: number;
  result?: Result;
  url?: string;
};

const pages: Page[] = [...document.querySelectorAll<HTMLElement>(".page")].map(
  (root) => {
    root.classList.add("is-empty");
    return {
      root,
      img: $<HTMLImageElement>("img", root),
      status: $(".status", root),
      input: $<HTMLInputElement>("input", root),
      revision: 0,
    };
  },
);

let worker: Worker | undefined;
let nextId = 0;
let active: { id: number; page: Page; revision: number } | undefined;
let pending: { page: Page; revision: number }[] = [];

function visibleBubbles(page: Page) {
  return (
    page.result?.bubbles.filter(
      (b) => b.score >= Number(threshold.value) / 100,
    ) ?? []
  );
}

function render(page: Page) {
  const svg = page.root.querySelector<SVGSVGElement>("svg")!;
  svg.replaceChildren();

  if (!page.result) {
    return;
  }

  const bubbles = visibleBubbles(page);
  const backendTag = page.result.backend === "webgpu" ? "WebGPU" : "WASM";
  page.status.textContent = `${bubbles.length} bubble${bubbles.length === 1 ? "" : "s"} · ${(page.result.milliseconds / 1000).toFixed(2)} s (${backendTag})${bubbles.length ? " · Click to zoom" : ""}`;
  svg.setAttribute("viewBox", `0 0 ${page.result.width} ${page.result.height}`);

  for (const [index, bubble] of bubbles.entries()) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", bubble.path);
    path.setAttribute("class", "bubble");
    path.setAttribute("fill-rule", "evenodd");
    path.setAttribute("tabindex", "0");
    path.setAttribute("role", "button");
    path.setAttribute(
      "aria-label",
      `Zoom bubble ${index + 1}, ${Math.round(bubble.score * 100)}% confidence`,
    );

    path.addEventListener("click", (e) => {
      e.stopPropagation();
      handleBubbleClick(page, bubble);
    });

    path.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        e.stopPropagation();
        handleBubbleClick(page, bubble);
      }
    });

    svg.append(path);

    if (confidence.checked) {
      const text = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "text",
      );
      text.setAttribute("x", String(bubble.box[0] + 5));
      text.setAttribute("y", String(bubble.box[1] + 23));
      text.setAttribute("class", "score");
      text.textContent = `${Math.round(bubble.score * 100)}%`;
      svg.append(text);
    }
  }
}

function handleBubbleClick(page: Page, bubble: Bubble) {
  if (isPopupVisible && activeBubble === bubble) {
    hideBubblePopup();
  } else {
    showBubblePopup(page, bubble);
  }
}

function renderBubble(page: Page, bubble: Bubble) {
  const [bx, by, bw, bh] = bubble.box;

  const pad = 3;
  const originX = Math.max(0, Math.floor(bx - pad));
  const originY = Math.max(0, Math.floor(by - pad));
  const maxRight = Math.min(page.img.naturalWidth, Math.ceil(bx + bw + pad));
  const maxBottom = Math.min(page.img.naturalHeight, Math.ceil(by + bh + pad));
  const cropW = Math.max(1, maxRight - originX);
  const cropH = Math.max(1, maxBottom - originY);

  const DEFAULT_ZOOM = 1.5;
  const MIN_WIDTH = Math.min(180, window.innerWidth * 0.8);
  const MAX_WIDTH = Math.min(480, window.innerWidth * 0.85);
  const MAX_HEIGHT = Math.min(540, window.innerHeight * 0.75);

  let displayW = cropW * DEFAULT_ZOOM;
  let displayH = cropH * DEFAULT_ZOOM;

  if (displayW < MIN_WIDTH) {
    const scale = MIN_WIDTH / displayW;
    displayW = MIN_WIDTH;
    displayH = displayH * scale;
  }

  if (displayW > MAX_WIDTH) {
    const scale = MAX_WIDTH / displayW;
    displayW = MAX_WIDTH;
    displayH = displayH * scale;
  }

  if (displayH > MAX_HEIGHT) {
    const scale = MAX_HEIGHT / displayH;
    displayH = MAX_HEIGHT;
    displayW = displayW * scale;
  }

  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  popupCanvas.width = Math.round(displayW * dpr);
  popupCanvas.height = Math.round(displayH * dpr);
  popupCanvas.style.width = `${Math.round(displayW)}px`;
  popupCanvas.style.height = `${Math.round(displayH)}px`;

  const ctx = popupCanvas.getContext("2d")!;
  ctx.clearRect(0, 0, popupCanvas.width, popupCanvas.height);

  ctx.save();
  const scaleX = popupCanvas.width / cropW;
  const scaleY = popupCanvas.height / cropH;
  ctx.scale(scaleX, scaleY);
  ctx.translate(-originX, -originY);

  const polygonPath = new Path2D(bubble.path);
  ctx.clip(polygonPath, "evenodd");

  ctx.drawImage(page.img, 0, 0);
  ctx.restore();
}

function updatePopupPosition(page: Page, bubble: Bubble) {
  const [bx, by, bw, bh] = bubble.box;

  const pad = 3;
  const originX = Math.max(0, Math.floor(bx - pad));
  const originY = Math.max(0, Math.floor(by - pad));
  const maxRight = Math.min(page.img.naturalWidth, Math.ceil(bx + bw + pad));
  const maxBottom = Math.min(page.img.naturalHeight, Math.ceil(by + bh + pad));
  const cropW = Math.max(1, maxRight - originX);
  const cropH = Math.max(1, maxBottom - originY);

  const imgRect = page.img.getBoundingClientRect();
  const scaleX = imgRect.width / page.img.naturalWidth;
  const scaleY = imgRect.height / page.img.naturalHeight;

  const cropCenterX = originX + cropW / 2;
  const cropCenterY = originY + cropH / 2;
  const screenCenterX = imgRect.left + cropCenterX * scaleX;
  const screenCenterY = imgRect.top + cropCenterY * scaleY;

  const displayW = parseFloat(popupCanvas.style.width) || cropW * 1.5;
  const displayH = parseFloat(popupCanvas.style.height) || cropH * 1.5;

  let left = screenCenterX - displayW / 2;
  let top = screenCenterY - displayH / 2;

  const margin = 12;
  left = Math.max(
    margin,
    Math.min(window.innerWidth - displayW - margin, left),
  );
  top = Math.max(margin, Math.min(window.innerHeight - displayH - margin, top));

  popup.style.setProperty("--x", `${Math.round(left)}px`);
  popup.style.setProperty("--y", `${Math.round(top)}px`);
}

function showBubblePopup(page: Page, bubble: Bubble) {
  clearTimeout(hideTimer);

  const isSwitching = isPopupVisible && activeBubble !== bubble;
  activeBubble = bubble;
  activePage = page;

  renderBubble(page, bubble);
  updatePopupPosition(page, bubble);

  if (isSwitching) {
    popup.classList.add("switching");

    requestAnimationFrame(() => {
      popup.classList.remove("switching");
      popup.classList.add("visible");
    });
  } else {
    popup.style.transition = "none";
    popup.hidden = false;
    void popup.offsetWidth;
    popup.style.transition = "";

    requestAnimationFrame(() => {
      isPopupVisible = true;
      popup.classList.add("visible");
    });
  }
}

function hideBubblePopup() {
  if (!isPopupVisible) {
    return;
  }

  isPopupVisible = false;
  activeBubble = undefined;
  activePage = undefined;

  popup.classList.remove("visible");
  popup.classList.remove("switching");

  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    if (!isPopupVisible) {
      popup.hidden = true;
    }
  }, 220);
}

function makeWorker() {
  const instance = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });

  instance.onmessage = ({ data }: MessageEvent<Reply>) => {
    if (!active || data.id !== active.id) {
      return;
    }

    const { page, revision } = active;

    if (page.revision === revision) {
      if (data.result) {
        page.result = data.result;

        if (webgpu.checked && data.result.backend === "wasm") {
          webgpu.checked = false;
          webgpu.disabled = true;
          webgpuRow.classList.add("disabled");
          webgpuRow.title = "WebGPU failed, falling back to WASM.";
          webgpuLabel.textContent = "WebGPU (unavailable)";
        }

        render(page);
      } else {
        page.status.textContent =
          data.error ?? "Detection failed. Drop an image to retry.";
      }
    }

    active = undefined;
    void pump();
  };

  instance.onerror = () => {
    if (active && active.page.revision === active.revision) {
      active.page.status.textContent =
        "Detection unavailable. Drop an image to retry.";
    }

    instance.terminate();
    worker = undefined;
    active = undefined;
    void pump();
  };

  return instance;
}

async function pump() {
  if (active) {
    return;
  }

  const job = pending.shift();
  if (!job) {
    return;
  }

  if (job.page.revision !== job.revision) {
    void pump();
    return;
  }

  const task = { ...job, id: ++nextId };
  active = task;

  try {
    const bitmap = await createImageBitmap(job.page.img);

    if (active !== task || job.page.revision !== job.revision) {
      bitmap.close();

      if (active === task) {
        active = undefined;
      }

      void pump();
      return;
    }

    await webgpuSupportedPromise;

    if (active !== task || job.page.revision !== job.revision) {
      bitmap.close();

      if (active === task) {
        active = undefined;
      }

      void pump();
      return;
    }

    worker ??= makeWorker();
    const backend: Backend =
      webgpu.checked && !webgpu.disabled ? "webgpu" : "wasm";
    worker.postMessage({ id: task.id, bitmap, backend }, [bitmap]);
  } catch {
    if (job.page.revision === job.revision) {
      job.page.status.textContent =
        "Could not read image. Select another image.";
    }

    if (active === task) {
      active = undefined;
    }

    void pump();
  }
}

function redetect(page: Page) {
  if (page.root.classList.contains("is-empty") || !page.img.naturalWidth) {
    return;
  }

  page.revision++;
  const revision = page.revision;
  page.result = undefined;
  page.status.textContent = "Detecting…";
  page.root.querySelector("svg")!.replaceChildren();
  hideBubblePopup();

  pending = pending.filter((job) => job.page !== page);
  pending.push({ page, revision });
  void pump();
}

function reset(page: Page) {
  page.revision++;
  page.result = undefined;

  if (page.url) {
    URL.revokeObjectURL(page.url);
  }

  page.url = undefined;
  page.img.removeAttribute("src");
  page.img.hidden = true;

  page.root.querySelector("svg")!.replaceChildren();
  $(".empty", page.root).hidden = false;
  page.root.classList.add("is-empty");
  page.status.textContent = "Empty";

  $(".sheet", page.root).style.removeProperty("--ratio");
  page.input.value = "";
}

async function load(page: Page, source: File | string) {
  reset(page);

  const revision = page.revision;
  const url = typeof source === "string" ? source : URL.createObjectURL(source);

  if (typeof source !== "string") {
    page.url = url;
  }

  page.status.textContent = "Detecting…";
  page.img.src = url;

  try {
    await page.img.decode();

    if (page.revision !== revision) {
      return;
    }

    page.img.hidden = false;
    $(".empty", page.root).hidden = true;
    page.root.classList.remove("is-empty");
    $(".sheet", page.root).style.setProperty(
      "--ratio",
      String(page.img.naturalWidth / page.img.naturalHeight),
    );

    pending = pending.filter((job) => job.page !== page);
    pending.push({ page, revision });
    void pump();
  } catch {
    if (page.revision === revision) {
      page.img.hidden = true;
      $(".empty", page.root).hidden = false;
      page.root.classList.add("is-empty");
      page.status.textContent = "Could not load image. Select another image.";
    }
  }
}

function loadFiles(files: FileList | File[], start = 0) {
  const images = [...files].filter(
    (file) =>
      file.type.startsWith("image/") ||
      /\.(png|jpe?g|webp|gif|avif|bmp)$/i.test(file.name),
  );

  if (!images.length) {
    pages[start].status.textContent = "Please choose an image file.";
    return;
  }

  const count = mobile.matches ? 1 : 2;
  images
    .slice(0, count)
    .forEach((file, i) => void load(pages[(start + i) % count], file));
}

pages.forEach((page, index) => {
  $(".select", page.root).addEventListener("click", () => {
    page.input.click();
  });

  page.input.addEventListener("change", () => {
    if (page.input.files?.length) {
      loadFiles(page.input.files, index);
    }
  });
});

// Listen on the document so dropping works over occupied pages and the margins.
document.addEventListener("dragover", (event) => {
  if (!event.dataTransfer?.types.includes("Files")) {
    return;
  }

  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";

  const target =
    (event.target as Element).closest<HTMLElement>(".page") ?? pages[0].root;
  pages.forEach((page) => {
    page.root.classList.toggle("dragging", page.root === target);
  });
});

document.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) {
    pages.forEach((page) => {
      page.root.classList.remove("dragging");
    });
  }
});

document.addEventListener("drop", (event) => {
  event.preventDefault();
  pages.forEach((page) => {
    page.root.classList.remove("dragging");
  });

  const target = (event.target as Element).closest<HTMLElement>(".page");
  if (event.dataTransfer?.files.length) {
    loadFiles(event.dataTransfer.files, Number(target?.dataset.page ?? 0));
  }
});

$("#clear").addEventListener("click", () => {
  worker?.terminate();
  worker = undefined;
  active = undefined;
  pending = [];

  hideBubblePopup();

  pages.forEach(reset);
});

function closeOptions() {
  options.hidden = true;
  optionsButton.setAttribute("aria-expanded", "false");
}

optionsButton.addEventListener("click", () => {
  options.hidden = !options.hidden;
  optionsButton.setAttribute("aria-expanded", String(!options.hidden));
});

popup.addEventListener("click", (event) => {
  event.stopPropagation();
  hideBubblePopup();
});

document.addEventListener("click", (event) => {
  const target = event.target as Node;

  if (!options.contains(target) && !optionsButton.contains(target)) {
    closeOptions();
  }

  if (isPopupVisible && !popup.contains(target)) {
    hideBubblePopup();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (isPopupVisible) {
      hideBubblePopup();
      return;
    }

    if (!options.hidden) {
      closeOptions();
      optionsButton.focus();
    }
  }
});

webgpu.addEventListener("change", () => {
  pages.forEach((page) => {
    redetect(page);
  });
});

outline.addEventListener("change", () => {
  book.classList.toggle("no-outline", !outline.checked);
});

confidence.addEventListener("change", () => {
  pages.forEach(render);
});

threshold.addEventListener("input", () => {
  $("#threshold-value").textContent = `${threshold.value}%`;
  pages.forEach(render);
});

window.addEventListener("resize", () => {
  if (isPopupVisible && activePage && activeBubble) {
    updatePopupPosition(activePage, activeBubble);
  }
});

window.addEventListener(
  "scroll",
  () => {
    if (isPopupVisible && activePage && activeBubble) {
      updatePopupPosition(activePage, activeBubble);
    }
  },
  { passive: true },
);

void load(pages[0], "./assets/page-1.jpg");

if (!mobile.matches) {
  void load(pages[1], "./assets/page-2.jpg");
} else {
  reset(pages[1]);
}
