# PodShorts Processor

Automatically create short, vertical video clips from a podcast video.

---

## How it works

1. Transcribes the podcast using WhisperX.
2. Finds Q&A or story moments with Google Gemini.
3. Cuts and processes clips into vertical videos focused on the main speaker.
4. Saves clips in `/output_shorts/`.

---

## Example Files

```
/input.mp4
/output_shorts/
    1.mp4
    2.mp4
    3.mp4
```
- **`input.mp4`**: Example podcast video (present in this repo).
- **`output_shorts/1.mp4`**, **`output_shorts/2.mp4`**, **`output_shorts/3.mp4`**: Example generated vertical shorts (present in this repo).

---

## See Example Videos

**Input Video:**  
[▶️ input.mp4](input.mp4)

**Generated Shorts:**  
[▶️ 1.mp4](output_shorts/short_1_1.03-60.02.mp4)  
[▶️ 2.mp4](output_shorts/short_2_61.73-102.56.mp4)  
[▶️ 3.mp4](output_shorts/short_3_292.29-333.59.mp4)

---

## Usage

1. Place your podcast video at `input.mp4` in the project folder.
2. Run:
   ```bash
   modal run process_video_on_modal.py --video-path ./input.mp4
   ```
3. Processed shorts will be in `/output_shorts/`.

---

## Requirements

- [Modal](https://modal.com/) account and CLI
- Add your Gemini API key as a Modal secret named `gemini-secret`
- Python dependencies in `requirements.txt` (handled inside the Modal container)

---

## Output

- After running, your video clips will be in `/output_shorts/` as:
  - `1.mp4`
  - `2.mp4`
  - `3.mp4`

You can play these files with any video player.

---
