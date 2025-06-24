# PodShorts Processor

Automatically create short, vertical video clips from a podcast video.

---

## How it works

1. Transcribes the podcast using WhisperX.
2. Finds Q&A or story moments with Google Gemini.
3. Cuts and processes clips into vertical videos focused on the main speaker.
4. Saves clips in `/output_shorts/`.

---

## Example Videos

You can see sample outputs below:

<iframe width="360" height="640" src="https://www.youtube.com/embed/-1SHfksjU1c" title="Demo Pratham Garg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<iframe width="360" height="640" src="https://www.youtube.com/embed/BhJaQxAMmmQ" title="Demo 2 Pratham Garg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Usage

1. Place your podcast video as `input.mp4` in the project folder.
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
