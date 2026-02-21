const form = document.getElementById("uploadForm");
const input = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const chooseBtn = document.getElementById("chooseBtn");
const preview = document.getElementById("preview");
const emptyPreview = document.getElementById("emptyPreview");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submitBtn");
const camHeatmap = document.getElementById("camHeatmap");
const camOverlay = document.getElementById("camOverlay");
const emptyCamHeatmap = document.getElementById("emptyCamHeatmap");
const emptyCamOverlay = document.getElementById("emptyCamOverlay");
let currentFile = null;

function resetResult() {
  labelEl.textContent = "-";
  labelEl.className = "label";
  confidenceEl.textContent = "-";
  statusEl.textContent = "";
  camHeatmap.removeAttribute("src");
  camOverlay.removeAttribute("src");
  camHeatmap.style.display = "none";
  camOverlay.style.display = "none";
  emptyCamHeatmap.style.display = "inline";
  emptyCamOverlay.style.display = "inline";
}

function showPreview(file) {
  if (!file) {
    preview.style.display = "none";
    emptyPreview.style.display = "inline";
    return;
  }
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  emptyPreview.style.display = "none";
}

function setFile(file) {
  resetResult();
  currentFile = file || null;
  showPreview(currentFile);
}

chooseBtn.addEventListener("click", () => {
  input.click();
});

dropZone.addEventListener("click", () => {
  input.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});

input.addEventListener("change", () => {
  const file = input.files && input.files[0];
  setFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("active");
  });
});

dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer && event.dataTransfer.files;
  if (!files || files.length === 0) {
    return;
  }
  const file = files[0];
  if (!file.type.startsWith("image/")) {
    statusEl.textContent = "Vui lòng chọn đúng tệp ảnh.";
    return;
  }
  setFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = currentFile;
  if (!file) {
    statusEl.textContent = "Vui lòng chọn ảnh trước khi phân tích.";
    return;
  }

  resetResult();
  submitBtn.disabled = true;
  statusEl.textContent = "Đang phân tích ảnh...";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict-with-cam", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Dự đoán thất bại.");
    }

    const rawLabel = String(data.label || "").toLowerCase();
    const mappedLabel =
      rawLabel === "real" || rawLabel === "that"
        ? "Ảnh thật"
        : (rawLabel === "fake" || rawLabel === "gia"
          ? "Ảnh AI"
          : (rawLabel === "uncertain" ? "Không chắc chắn" : data.label));
    const confidencePct = (Number(data.confidence) * 100).toFixed(2) + "%";

    labelEl.textContent = mappedLabel;
    labelEl.className =
      "label " +
      ((rawLabel === "real" || rawLabel === "that")
        ? "real"
        : ((rawLabel === "fake" || rawLabel === "gia")
          ? "fake"
          : (rawLabel === "uncertain" ? "uncertain" : "")));
    confidenceEl.textContent = confidencePct;
    if (data.cam_heatmap_base64) {
      camHeatmap.src = "data:image/png;base64," + data.cam_heatmap_base64;
      camHeatmap.style.display = "block";
      emptyCamHeatmap.style.display = "none";
    }
    if (data.cam_overlay_base64) {
      camOverlay.src = "data:image/png;base64," + data.cam_overlay_base64;
      camOverlay.style.display = "block";
      emptyCamOverlay.style.display = "none";
    }
    statusEl.textContent =
      rawLabel === "uncertain"
        ? "Kết quả chưa chắc chắn. Bạn nên dùng ảnh rõ hơn hoặc kiểm tra thêm."
        : "Phân tích hoàn tất.";
  } catch (error) {
    statusEl.textContent = "Lỗi: " + error.message;
  } finally {
    submitBtn.disabled = false;
  }
});

resetResult();
