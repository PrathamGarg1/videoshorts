import glob
import json
import logging
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import modal


def setup_logger(name: str = "podshorts") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = setup_logger()

image = (modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.12")
         .apt_install(["ffmpeg","libgl1-mesa-glx","wget","libcudnn8","libcudnn8-dev"])
         .pip_install_from_requirements("requirements.txt")
          .add_local_dir("asd","/asd",copy=True)
         )
app = modal.App("podshorts-processor", image=image)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .add_local_file("index.html", "/root/index.html")
)

model_volume = modal.Volume.from_name("podshorts-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("podshorts-output", create_if_missing=True)

MODEL_CACHE_PATH = "/root/.cache/torch"
OUTPUT_DIR_PATH = "/output"
JOBS_DIR = pathlib.Path(OUTPUT_DIR_PATH) / "jobs"

# Free-tier Flash models only (Google AI Studio). Override via GEMINI_MODEL env var.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def job_status_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job_status(job_id: str, **fields):
    """Write/merge job status JSON on the output volume and commit."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = job_status_path(job_id)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    data.update(fields)
    data["job_id"] = job_id
    data["updated_at"] = time.time()
    path.write_text(json.dumps(data))
    output_volume.commit()
    log.info(f"[{job_id}] status → {data.get('status')} ({data.get('message', '')})")


def read_job_status(job_id: str) -> dict | None:
    path = job_status_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_cmd(label: str, command: str, *, cwd: str = None) -> subprocess.CompletedProcess:
    log.info(f"[cmd] {label}: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        log.error(f"[cmd] {label} failed (exit {result.returncode})")
        if result.stderr:
            log.error(f"[cmd] stderr: {result.stderr[-2000:]}")
        if result.stdout:
            log.error(f"[cmd] stdout: {result.stdout[-1000:]}")
        result.check_returncode()
    return result


def create_vertical_shorts(tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_vid_path):
    import cv2
    import ffmpegcv
    import numpy as np
    from tqdm import tqdm

    frame_rate = 25
    target_width = 1080
    target_height = 1920

    framelist = sorted(glob.glob(os.path.join(pyframes_path, "*.jpg")))
    faces = [[] for _ in range(len(framelist))]

    for trackidx, track in enumerate(tracks):
        scores_for_personi = scores[trackidx]
        for frameidx, frame in enumerate(track["track"]["frame"].tolist()):
            score_around_frameidx = scores_for_personi[max(frameidx - 30, 0):min(frameidx + 30, len(scores_for_personi))]
            avg_score = float(np.mean(score_around_frameidx) if len(score_around_frameidx) > 0 else 0)
            faces[frame].append({'track': trackidx, 'score': avg_score, 's': track['proc_track']["s"][frameidx], 'x': track['proc_track']["x"][frameidx], 'y': track['proc_track']["y"][frameidx]})

    tmp_video_path = os.path.join(pyavi_path, "video_only.mp4")
    vout = None

    for frameidx, framename in tqdm(enumerate(framelist), desc="Creating vertical short"):
        img = cv2.imread(framename)
        if img is None: continue

        current_faces = faces[frameidx]
        max_score_face = max(current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(file=tmp_video_path, codec=None, fps=frame_rate, resize=(target_width, target_height))
        
        mode = "crop" if max_score_face else "fullvdo"

        if mode == "fullvdo":
            background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_img = cv2.resize(img, (target_width, resized_height))
            center_y = (target_height - resized_height) // 2
            if center_y >= 0:
                background[center_y:center_y + resized_height, :] = resized_img
            vout.write(background)
        else:
            scale = target_height / img.shape[0]
            resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_img.shape[1]
            center_x = int(max_score_face["x"] * scale)
            left_x = max(min(center_x - target_width // 2, frame_width - target_width), 0)
            image_cropped = resized_img[0:target_height, left_x: left_x + target_width]
            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {tmp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{vertical_vid_path}")
    run_cmd("merge vertical video + audio", ffmpeg_command)


def process_clip(base_tmp_dir, original_video_path, start_time, end_time, short_index, job_id: str = ""):
    tag = f"[{job_id}] clip {short_index + 1}" if job_id else f"clip {short_index + 1}"
    log.info(f"{tag} | cutting {start_time:.1f}s → {end_time:.1f}s")
    short_name = f"short_{short_index}"
    short_tmp_dir = base_tmp_dir / short_name
    short_tmp_dir.mkdir(parents=True, exist_ok=True)

    short_segment_path = short_tmp_dir / f"{short_name}_segment.mp4"
    pywork_path = short_tmp_dir / "pywork"
    pyframes_path = short_tmp_dir / "pyframes"
    pyavi_path = short_tmp_dir / "pyavi"
    pywork_path.mkdir(exist_ok=True, parents=True)
    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    vertical_vid_path = pyavi_path / f"{short_name}_final_vertical.mp4"
    audio_path = pyavi_path / "audio.wav"

    durationofvdo = end_time - start_time
    cut_command = f"ffmpeg -i {original_video_path} -ss {start_time} -t {durationofvdo} -y {short_segment_path}"
    run_cmd(f"{tag} cut segment", cut_command)

    extractcmd = f"ffmpeg -i {short_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y {audio_path}"
    run_cmd(f"{tag} extract audio", extractcmd)

    shutil.copy(short_segment_path, base_tmp_dir / f"{short_name}.mp4")

    lrasd_command = f"python Columbia_test.py --videoName {short_name} --videoFolder {str(base_tmp_dir)} --pretrainModel weight/finetuning_TalkSet.model"
    log.info(f"{tag} | running active speaker detection (LR-ASD)")
    try:
        run_cmd(f"{tag} LR-ASD", lrasd_command, cwd="/asd")
    except subprocess.CalledProcessError:
        asd_files = list(pathlib.Path("/asd").glob("*")) if pathlib.Path("/asd").exists() else []
        log.error(f"{tag} | /asd contents: {[f.name for f in asd_files[:20]]}")
        raise

    tracks_path = pywork_path / "tracks.pckl"
    scores_path = pywork_path / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"{tag} | Speaker detection output missing (tracks.pckl or scores.pckl)")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    log.info(f"{tag} | rendering vertical 1080x1920")
    create_vertical_shorts(tracks, scores, str(pyframes_path), str(pyavi_path), str(audio_path), str(vertical_vid_path))
    log.info(f"{tag} | done")
    
    return vertical_vid_path


@app.cls(
    gpu="L40S", 
    timeout=1800, 
    secrets=[modal.Secret.from_name("gemini-secret")],
    volumes={MODEL_CACHE_PATH: model_volume, OUTPUT_DIR_PATH: output_volume},
)
class VideoProcessor:
    @modal.enter()
    def load_models(self):
        """This method runs once when the container starts up to load the models."""
        import whisperx
        from google import genai

        log.info("GPU container starting — loading models...")
        log.info(f"Gemini model: {GEMINI_MODEL}")
        self.device = "cuda"
        self.compute_type = "float16"

        t0 = time.time()
        self.whisper_model = whisperx.load_model("large-v2", device=self.device, compute_type=self.compute_type)
        log.info(f"WhisperX loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code="en", device=self.device)
        log.info(f"Alignment model loaded in {time.time() - t0:.1f}s")

        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        asd_path = pathlib.Path("/asd")
        if asd_path.exists() and (asd_path / "Columbia_test.py").exists():
            log.info("LR-ASD ready at /asd (Columbia_test.py + weights found)")
        else:
            log.error("LR-ASD missing at /asd — clip generation will fail")
        log.info("All models ready.")
        model_volume.commit()

    def _transcribe(self, base_dir, videopath):
        import whisperx

        audio_path = base_dir / "audio.wav"
        extractcmd = f"ffmpeg -i {videopath} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        run_cmd("extract audio for transcription", extractcmd)

        log.info("Transcribing with WhisperX...")
        t0 = time.time()
        audio = whisperx.load_audio(str(audio_path))
        resultinit = self.whisper_model.transcribe(audio, batch_size=16)
        result = whisperx.align(resultinit["segments"], self.align_model, self.align_metadata, audio, device=self.device, return_char_alignments=False)
        word_count = len(result.get("word_segments", []))
        log.info(f"Transcription done in {time.time() - t0:.1f}s — {word_count} words")
        
        return json.dumps([
            {"start": seg["start"], "end": seg["end"], "word": seg["word"]}
            for seg in result["word_segments"]
        ])

    def _identify_moments(self, transcript_json):
        prompt = """
This is a podcast video transcript consisting of words, along with each word's start and end time. Create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.
Your task is to find and extract stories, or a question and its corresponding answer from the transcript. Make sure the clip is viral and has a hook in the beginning for the user to stay in the video. Each clip should begin with the question and conclude with the answer. It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.
Please adhere to the following rules:
- Ensure that clips do not overlap with one another.
- Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
- Only use the start and end timestamps provided in the input. Modifying timestamps is not allowed.
- Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...]. The output must always be readable by Python's json.loads function.
- Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.
Never include:
- Moments of greeting, thanking, or saying goodbye.
- Non-question and answer interactions.
If there are no valid clips to extract, the output should be an empty list [], in JSON format.
The transcript is as follows:\n\n""" + transcript_json
        
        log.info(f"Calling Gemini ({GEMINI_MODEL}) to identify clip moments...")
        t0 = time.time()
        try:
            response = self.gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            log.info(f"Gemini responded in {time.time() - t0:.1f}s")
            log.info(f"Gemini raw response (first 500 chars): {response.text[:500]}")
            return response.text
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            log.error(traceback.format_exc())
            raise

    @modal.method()
    def process_video(self, video_bytes: bytes, job_id: str = None):
        """The main remote method that receives the video and processes it."""
        if job_id is None:
            job_id = str(uuid.uuid4())[:8]

        size_mb = len(video_bytes) / (1024 * 1024)
        log.info(f"[{job_id}] Pipeline started — video size: {size_mb:.1f} MB")
        pipeline_start = time.time()
        write_job_status(job_id, status="processing", message="Starting pipeline", clips=[])

        base_dir = pathlib.Path("/tmp") / str(uuid.uuid4())
        base_dir.mkdir()

        video_path_in_container = base_dir / "input.mp4"
        video_path_in_container.write_bytes(video_bytes)
        log.info(f"[{job_id}] Video saved to {video_path_in_container}")

        try:
            write_job_status(job_id, status="processing", message="Step 1/3 — Transcription")
            log.info(f"[{job_id}] Step 1/3 — Transcription")
            transcripts_json = self._transcribe(base_dir, video_path_in_container)

            write_job_status(job_id, status="processing", message="Step 2/3 — Moment detection")
            log.info(f"[{job_id}] Step 2/3 — Moment detection (Gemini)")
            moments_response = self._identify_moments(transcripts_json)

            moments = moments_response.strip()
            if moments.startswith("```json"):
                moments = moments[len("```json"):].strip()
            if moments.endswith("```"):
                moments = moments[:-len("```")].strip()

            try:
                moments = json.loads(moments)
                if not moments or not isinstance(moments, list):
                    log.warning(f"[{job_id}] Gemini returned empty or invalid list")
                    moments = []
            except json.JSONDecodeError:
                log.error(f"[{job_id}] Failed to parse Gemini JSON: {moments[:300]}")
                moments = []

            log.info(f"[{job_id}] Found {len(moments)} clip(s): {moments}")
            write_job_status(
                job_id,
                status="processing",
                message=f"Step 3/3 — Generating {len(moments)} clip(s)",
                moments_found=len(moments),
            )

            log.info(f"[{job_id}] Step 3/3 — Generating vertical clips")
            clip_filenames = []
            for idx, moment in enumerate(moments):
                if "start" in moment and "end" in moment:
                    try:
                        write_job_status(
                            job_id,
                            status="processing",
                            message=f"Rendering clip {idx + 1}/{len(moments)}",
                            clips=[{"name": f, "url": f"/clips/{f}"} for f in clip_filenames],
                        )
                        final_clip_path = process_clip(
                            base_dir, video_path_in_container,
                            moment["start"], moment["end"], idx, job_id=job_id,
                        )
                        output_filename = f"{job_id}_short_{idx+1}.mp4"
                        destination_path = pathlib.Path(OUTPUT_DIR_PATH) / output_filename
                        shutil.copy(final_clip_path, destination_path)
                        clip_filenames.append(output_filename)
                        output_volume.commit()
                        log.info(f"[{job_id}] Saved {output_filename}")
                    except Exception as e:
                        log.error(f"[{job_id}] Clip {idx + 1} failed: {e}")
                        log.error(traceback.format_exc())

            elapsed = time.time() - pipeline_start
            clips_payload = [{"name": f, "url": f"/clips/{f}"} for f in clip_filenames]
            write_job_status(
                job_id,
                status="done",
                message=f"Done — {len(clip_filenames)} clip(s) in {elapsed:.0f}s",
                clips=clips_payload,
                count=len(clip_filenames),
            )
            log.info(f"[{job_id}] Pipeline finished in {elapsed:.1f}s — {len(clip_filenames)} clip(s) produced")
            return clip_filenames

        except Exception as e:
            log.error(f"[{job_id}] Pipeline failed: {e}")
            log.error(traceback.format_exc())
            write_job_status(job_id, status="error", message=str(e), error=str(e))
            raise
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


@app.function(image=web_image, volumes={OUTPUT_DIR_PATH: output_volume}, timeout=600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response

    web_app = FastAPI(title="Video Shorts")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    web_log = setup_logger("podshorts-web")

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    @web_app.get("/")
    async def serve_ui():
        web_log.info("Serving UI")
        return FileResponse("/root/index.html")

    @web_app.post("/process")
    async def process_upload(file: UploadFile = File(...)):
        job_id = str(uuid.uuid4())[:8]
        web_log.info(f"[{job_id}] Upload received — filename: {file.filename}, content_type: {file.content_type}")

        t0 = time.time()
        video_bytes = await file.read()
        size_mb = len(video_bytes) / (1024 * 1024)
        web_log.info(f"[{job_id}] Read {size_mb:.1f} MB in {time.time() - t0:.1f}s")

        if size_mb < 0.01:
            return JSONResponse({"detail": "Empty or invalid video file", "job_id": job_id}, status_code=400)

        write_job_status(job_id, status="queued", message="Queued — GPU worker starting", clips=[])
        web_log.info(f"[{job_id}] Spawning GPU worker (async — client can disconnect)")

        try:
            processor = VideoProcessor()
            processor.process_video.spawn(video_bytes, job_id=job_id)
        except Exception as e:
            web_log.error(f"[{job_id}] Failed to spawn GPU worker: {e}")
            web_log.error(traceback.format_exc())
            write_job_status(job_id, status="error", message=str(e), error=str(e))
            return JSONResponse({"detail": f"Failed to start processing: {e}", "job_id": job_id}, status_code=500)

        return JSONResponse(
            {"job_id": job_id, "status": "queued", "message": "Processing started"},
            status_code=202,
        )

    @web_app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        if not job_id or "/" in job_id or ".." in job_id or len(job_id) > 64:
            raise HTTPException(status_code=400, detail="Invalid job id")

        output_volume.reload()
        data = read_job_status(job_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Job not found")

        clips = data.get("clips") or []
        # Ensure clip URLs are present for older status files
        for c in clips:
            if isinstance(c, str):
                continue
            if "url" not in c and "name" in c:
                c["url"] = f"/clips/{c['name']}"

        return {
            "job_id": job_id,
            "status": data.get("status", "unknown"),
            "message": data.get("message", ""),
            "clips": clips,
            "count": data.get("count", len(clips) if isinstance(clips, list) else 0),
            "error": data.get("error"),
            "updated_at": data.get("updated_at"),
        }

    @web_app.get("/clips/{filename}")
    async def serve_clip(filename: str):
        if not filename.endswith(".mp4") or "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid clip name")

        output_volume.reload()
        try:
            data = b"".join(output_volume.read_file(filename))
        except Exception:
            web_log.error(f"Clip not found: {filename}")
            raise HTTPException(status_code=404, detail="Clip not found")

        web_log.info(f"Serving clip {filename} ({len(data) / (1024 * 1024):.1f} MB)")
        return Response(
            content=data,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    return web_app


@app.local_entrypoint()
def main(video_path: str = "./input.mp4"):
    if not os.path.exists(video_path):
        print(" Video  not found  ")
        return

    print(f"Reading video file from: {video_path}")
    video_bytes = pathlib.Path(video_path).read_bytes()

    print("Uploading video and starting processing...")
    processor = VideoProcessor()
    processed_filenames = processor.process_video.remote(video_bytes)

    if not processed_filenames:
        print("No clips were generated.")
        return

    local_output_dir = pathlib.Path("./output_shorts")
    local_output_dir.mkdir(exist_ok=True)
    
    print("Downloading clips...")

    for filename in processed_filenames:
        data = b"".join(output_volume.read_file(filename))
        local_path = local_output_dir / filename
        local_path.write_bytes(data)
        print(f" Saved {local_path}")
        
    print("\n Made with your blessings")

