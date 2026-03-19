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
const historyList = document.getElementById("historyList");
const historyImage = document.getElementById("historyImage");
const historyEmpty = document.getElementById("historyEmpty");
const historyMeta = document.getElementById("historyMeta");
const historyStatus = document.getElementById("historyStatus");
const refreshHistoryBtn = document.getElementById("refreshHistory");
let currentFile = null;
let activeHistoryId = null;

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

function clearHistoryPreview() {
  historyImage.removeAttribute("src");
  historyImage.style.display = "none";
  historyEmpty.style.display = "inline";
  historyMeta.textContent = "";
  activeHistoryId = null;
  const activeItem = historyList.querySelector(".history-item.active");
  if (activeItem) {
    activeItem.classList.remove("active");
  }
}

function renderHistoryItems(items) {
  historyList.textContent = "";
  if (!items || items.length === 0) {
    const empty = document.createElement("li");
    empty.className = "history-item";
    empty.textContent = "Chưa có lịch sử.";
    historyList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.dataset.id = String(item.id);

    const title = document.createElement("div");
    title.className = "history-item-title";
    title.textContent = item.filename || "Không tên";

    const meta = document.createElement("div");
    meta.className = "history-item-meta";
    meta.textContent = `${item.created_at} | ${item.label} | ${(Number(item.confidence) * 100).toFixed(2)}%`;

    li.appendChild(title);
    li.appendChild(meta);
    li.addEventListener("click", () => loadHistoryDetail(item.id, li));
    historyList.appendChild(li);
  });
}

async function loadHistory() {
  if (!historyList) {
    return;
  }
  historyStatus.textContent = "Đang tải lịch sử...";
  try {
    const response = await fetch("/history?limit=50");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Không tải được lịch sử.");
    }
    renderHistoryItems(data);
    historyStatus.textContent = `Đã tải ${data.length} mục.`;
  } catch (error) {
    historyStatus.textContent = "Lỗi: " + error.message;
  }
}

async function loadHistoryDetail(itemId, element) {
  if (!itemId) {
    return;
  }
  if (activeHistoryId === itemId) {
    return;
  }
  historyStatus.textContent = "Đang tải chi tiết...";
  try {
    const response = await fetch(`/history/${itemId}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Không tải được chi tiết.");
    }

    historyImage.src = `data:${data.image_mime};base64,${data.image_base64}`;
    historyImage.style.display = "block";
    historyEmpty.style.display = "none";

    historyMeta.innerHTML = `
      <div><strong>Thời gian:</strong> ${data.created_at}</div>
      <div><strong>Tên file:</strong> ${data.filename}</div>
      <div><strong>Kết quả:</strong> ${data.label}</div>
      <div><strong>Độ tin cậy:</strong> ${(Number(data.confidence) * 100).toFixed(2)}%</div>
    `;

    const activeItem = historyList.querySelector(".history-item.active");
    if (activeItem) {
      activeItem.classList.remove("active");
    }
    element.classList.add("active");
    activeHistoryId = itemId;
    historyStatus.textContent = "Đã tải chi tiết.";
  } catch (error) {
    historyStatus.textContent = "Lỗi: " + error.message;
  }
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
clearHistoryPreview();
loadHistory();

if (refreshHistoryBtn) {
  refreshHistoryBtn.addEventListener("click", () => {
    clearHistoryPreview();
    loadHistory();
  });
}
