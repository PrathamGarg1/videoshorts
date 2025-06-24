# AI Video Shorts Generator

A GPU-accelerated pipeline to automatically create vertical video shorts from podcasts or long videos.

---

## Description

**AI Video Shorts Generator** uses Python, Modal, WhisperX, Gemini API, and Active Speaker Detection (LR-ASD, IJCV 2025) to:
- Transcribe videos and accurately segment speech.
- Detect active speakers and keep the vertical frame focused on them.
- Automatically extract key moments (Q&A, stories, viral hooks) using Gemini API.
- Generate ready-to-upload vertical video shorts.

---

## Features

- **Transcription:** Uses WhisperX for accurate speech-to-text.
- **Moment Detection:** Finds questions, answers, and stories using Gemini API.
- **Active Speaker Detection:** Keeps the shot centered on whoever is speaking (LR-ASD model).
- **GPU-Accelerated:** Fast processing with Modal cloud GPUs.
- **Automated End-to-End:** Input a video, get vertical shorts as output.


---

## Demo Output


You can watch sample output shorts here:
- [Demo Short 1](https://www.youtube.com/shorts/-1SHfksjU1c)
- [Demo Short 2](https://www.youtube.com/shorts/BhJaQxAMmmQ)
- [Demo Short 3](https://www.youtube.com/shorts/BhJaQxAMmmQ)

The input video used:
- [Input Video](https://youtu.be/P6FjXQxs7bQ)

---

---


## Usage

1. Place your podcast or long video as `input.mp4` in the project directory.
2. Run:
   ```bash
   modal run process_video_on_modal.py --video-path ./input.mp4
   ```
3. Find the generated vertical shorts in `/output_shorts/` (e.g., `1.mp4`, `2.mp4`, ...).

---

## Requirements

- Modal account and CLI (`pip install modal`)
- Add your Gemini API key as a Modal secret named `gemini-secret`
- Python dependencies in `requirements.txt` (auto-installed by Modal container)


## Credits

- WhisperX for transcription
- Google Gemini API for moment extraction
- LR-ASD (Springer IJCV 2025) for active speaker detection
- Modal for GPU orchestration

---
