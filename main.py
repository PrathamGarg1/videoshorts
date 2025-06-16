# process_video_on_modal.py

import glob
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import uuid

import modal

# --- 1. Define the Modal Environment ---
# This section defines the container image where our code will run.
# It includes Python libraries and the necessary 'lrasd' directory.
image = (modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.12")
         .apt_install(["ffmpeg","libgl1-mesa-glx","wget","libcudnn8","libcudnn8-dev"])
         .pip_install_from_requirements("requirements.txt")
          .add_local_dir("lrasd","/asd",copy=True)  # since made some chnages therefore copy from locall
         )
app = modal.App("podshorts-processor", image=image)

# --- 2. Define Persistent Volumes ---
# A 'Volume' is like a network hard drive.
# - model_volume caches the AI models to avoid re-downloading them on every run.
# - output_volume stores the generated video clips before they are saved locally.
model_volume = modal.Volume.from_name("podshorts-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("podshorts-output", create_if_missing=True)

MODEL_CACHE_PATH = "/root/.cache/torch"
OUTPUT_DIR_PATH = "/output"


# --- 3. Core Logic (Helper Functions) ---
# These functions are copied from your original script and will run inside the Modal container.

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
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True, capture_output=True)


def process_clip(base_tmp_dir, original_video_path, start_time, end_time, short_index):
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
    subprocess.run(cut_command, shell=True, check=True, capture_output=True)

    extractcmd = f"ffmpeg -i {short_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y {audio_path}"
    subprocess.run(extractcmd, shell=True, check=True, capture_output=True)

    shutil.copy(short_segment_path, base_tmp_dir / f"{short_name}.mp4")

    # Run Active Speaker Detection. Note the corrected 'cwd' path.
    lrasd_command = f"python Columbia_test.py --videoName {short_name} --videoFolder {str(base_tmp_dir)} --pretrainModel weight/finetuning_TalkSet.model"
    subprocess.run(lrasd_command, cwd="/asd", shell=True, check=True, capture_output=True, text=True)

    tracks_path = pywork_path / "tracks.pckl"
    scores_path = pywork_path / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Speaker detection output files (tracks.pckl or scores.pckl) not found.")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    create_vertical_shorts(tracks, scores, str(pyframes_path), str(pyavi_path), str(audio_path), str(vertical_vid_path))
    
    # Return the path of the final generated clip inside the container
    return vertical_vid_path


# --- 4. The Remote GPU-accelerated Class ---
# This class contains the main logic that will run on a Modal GPU.
@app.cls(
    gpu="L40S",  # Specify GPU type. "T4" is a cheaper alternative.
    timeout=1800, # 30-minute timeout for the whole process.
    secrets=[modal.Secret.from_name("gemini-secret")],
    volumes={MODEL_CACHE_PATH: model_volume, OUTPUT_DIR_PATH: output_volume},
)
class VideoProcessor:
    @modal.enter()
    def load_models(self):
        """This method runs once when the container starts up to load the models."""
        import whisperx
        from google import genai
        
        print("Loading AI models onto GPU...")
        self.device = "cuda"
        self.compute_type = "float16"

        self.whisper_model = whisperx.load_model("large-v2", device=self.device, compute_type=self.compute_type)
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code="en", device=self.device)
        
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        
        print("Models loaded successfully.")
        model_volume.commit() # Save downloaded models to the volume.

    def _transcribe(self, base_dir, videopath):
        import whisperx

        audio_path = base_dir / "audio.wav"
        extractcmd = f"ffmpeg -i {videopath} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extractcmd, shell=True, check=True, capture_output=True)

        print("Starting transcription...")
        audio = whisperx.load_audio(str(audio_path))
        resultinit = self.whisper_model.transcribe(audio, batch_size=16)
        result = whisperx.align(resultinit["segments"], self.align_model, self.align_metadata, audio, device=self.device, return_char_alignments=False)
        
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
        
        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents=prompt)
        print(f"Identified moments response: {response.text}")
        return response.text

    @modal.method()
    def process_video(self, video_bytes: bytes):
        """The main remote method that receives the video and processes it."""
        print(f"Received video ({len(video_bytes) / 1e6:.2f} MB). Starting pipeline...")
        base_dir = pathlib.Path("/tmp") / str(uuid.uuid4())
        base_dir.mkdir()
        
        video_path_in_container = base_dir / "input.mp4"
        video_path_in_container.write_bytes(video_bytes)

        # 1. Transcribe video
        transcripts_json = self._transcribe(base_dir, video_path_in_container)

        # 2. Identify key moments with Gemini
        moments_response = self._identify_moments(transcripts_json)
        try:
            # moments_str = moments_response.strip().replace("``````", "").strip()
            # moments = json.loads(moments_str)
            # if not isinstance(moments, list): moments = []

            moments = moments_response.strip()
            if moments.startswith("```json"):
                moments = moments[len("```json"):].strip()
            if moments.endswith("```"):
                moments = moments[:-len("```")].strip()
            
        
            print("here1")
            print(moments)

            moments = json.loads(moments)
            print("here2")
            print(moments)

            if not moments or not isinstance(moments,list):
                print("The returned is not a list")
                moments = []

            print("clips are : -")
            print(moments)
            print("😀😀")

        except json.JSONDecodeError:
            print("Error: Could not parse moments from Gemini response.")
            moments = []



        print(f"Found {len(moments)} potential clips to generate.")

        # 3. Process each clip and save to output volume
        clip_filenames = []
        for idx, moment in enumerate(moments):
            if "start" in moment and "end" in moment:
                print(f"\n--- Processing clip {idx + 1}/{len(moments)} (Start: {moment['start']}s, End: {moment['end']}s) ---")
                try:
                    final_clip_path = process_clip(base_dir, video_path_in_container, moment["start"], moment["end"], idx)
                    
                    output_filename = f"short_{idx+1}_{moment['start']:.2f}-{moment['end']:.2f}.mp4"
                    destination_path = pathlib.Path(OUTPUT_DIR_PATH) / output_filename
                    
                    shutil.copy(final_clip_path, destination_path)
                    
                    clip_filenames.append(output_filename)
                    print(f"Successfully processed and saved clip to volume: {destination_path}")
                except Exception as e:
                    print(f"Failed to process clip {idx+1}. Error: {e}")

        output_volume.commit()
        shutil.rmtree(base_dir)
        return clip_filenames

# --- 5. The Local Entrypoint ---
@app.local_entrypoint()
def main(video_path: str = "./input.mp4"):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'.")
        print("Please provide the correct path using --video-path /path/to/your/video.mp4")
        return

    print(f"Reading video file from: {video_path}")
    video_bytes = pathlib.Path(video_path).read_bytes()

    print("Uploading video and starting remote processing on Modal GPU...")
    processor = VideoProcessor()
    processed_filenames = processor.process_video.remote(video_bytes)

    if not processed_filenames:
        print("\nProcessing complete. No clips were generated.")
        return

    # Create local output directory
    local_output_dir = pathlib.Path("./output_shorts")
    local_output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing complete. Downloading {len(processed_filenames)} clips to '{local_output_dir.resolve()}'...")

    # Download each processed clip from the output volume
    for filename in processed_filenames:
        data = b"".join(output_volume.read_file(filename))
        local_path = local_output_dir / filename
        local_path.write_bytes(data)
        print(f"-> Saved {local_path}")
        
    print("\nAll clips saved successfully.")

