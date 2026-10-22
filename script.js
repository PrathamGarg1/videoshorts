const fileInput = document.getElementById("file");
const statusEl = document.getElementById("status");
const fileNameEl = document.getElementById("fileName");
const clipsEl = document.getElementById("clips");
const clipTemplate = document.getElementById("clip-template");
const dropzone = document.getElementById("dropzone");

let pollTimer = null;

fileInput.onchange = () => upload(fileInput.files[0]);

dropzone.onclick = () => {
  if (dropzone.getAttribute("aria-disabled") === "true") return;
  fileInput.click();
};
dropzone.onkeydown = (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  if (dropzone.getAttribute("aria-disabled") === "true") return;
  fileInput.click();
};
dropzone.ondragenter = (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
};
dropzone.ondragover = (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
};
dropzone.ondragleave = () => dropzone.classList.remove("dragover");
dropzone.ondrop = (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  upload(event.dataTransfer.files[0]);
};

async function upload(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|mkv)$/i.test(file.name)) {
    setStatus("Use a video file: MP4, MOV, or WebM.", "error");
    fileInput.value = "";
    return;
  }

  clearTimeout(pollTimer);
  clipsEl.innerHTML = "";
  fileNameEl.textContent = file.name + " · " + formatSize(file.size);
  setStatus("Uploading…", "processing");

  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/process", { method: "POST", body: form });
    const data = await res.json();
    if (!data.job_id) throw new Error(data.detail || "Upload failed");
    watch(data.job_id);
  } catch (err) {
    setStatus(err.message || "Something went wrong", "error");
  }
  fileInput.value = "";
}

function watch(jobId) {
  clearTimeout(pollTimer);
  localStorage.setItem("podshorts_job_id", jobId);

  async function tick() {
    try {
      const res = await fetch("/jobs/" + jobId);
      if (res.status === 404) {
        setStatus("That job is not on the server. Drop the episode again.", "error");
        localStorage.removeItem("podshorts_job_id");
        return;
      }
      const job = await res.json();
      const stillWorking = job.status !== "done" && job.status !== "error";
      setStatus(job.message || job.status, stillWorking ? "processing" : job.status);

      if (job.status === "done") {
        showClips(job.clips || []);
        localStorage.removeItem("podshorts_job_id");
        return;
      }
      if (job.status === "error") {
        setStatus(job.error || job.message, "error");
        localStorage.removeItem("podshorts_job_id");
        return;
      }
      if (job.clips && job.clips.length) showClips(job.clips);
      pollTimer = setTimeout(tick, 5000);
    } catch (err) {
      setStatus("Still waiting…", "processing");
      pollTimer = setTimeout(tick, 5000);
    }
  }

  tick();
}

function showClips(clips) {
  clipsEl.innerHTML = "";
  let number = 1;
  for (const clip of clips) {
    const card = clipTemplate.content.cloneNode(true);
    const url = clip.url || ("/clips/" + (clip.name || clip));
    card.querySelector("video").src = url;
    card.querySelector("span").textContent = "Clip " + String(number).padStart(2, "0");
    const link = card.querySelector("a");
    link.href = url;
    link.download = clip.name || clip;
    clipsEl.appendChild(card);
    number += 1;
  }
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
  document.body.classList.toggle("is-live", kind === "processing");
  dropzone.setAttribute("aria-disabled", kind === "processing" ? "true" : "false");
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

const savedJob = localStorage.getItem("podshorts_job_id");
if (savedJob) watch(savedJob);
