# FrameTrace

Find where a person appears in a video, using one reference photo.

## What it does

Upload a photo of the person you are looking for and a video. FrameTrace scans the video and reports the timestamp where that face first appears.

## How it works

1. Detect the face in the reference photo and turn it into a 128-number embedding with the `face_recognition` library (built on dlib).
2. Read the video with OpenCV. To keep it fast, process every 5th frame and shrink each frame to 25% of its size.
3. Detect and embed every face in each sampled frame.
4. Compare each face to the reference embedding. If the distance is within the tolerance (0.6), it counts as a match.
5. Report the match time as frame number divided by the video's FPS.

## Tech stack

Python, OpenCV, face_recognition (dlib), Streamlit

## Run it locally

```bash
git clone https://github.com/talhabutt72/FrameTrace.git
cd FrameTrace
pip install -r requirements.txt
streamlit run app.py
```

Note: `face_recognition` depends on dlib, which needs CMake and a C++ compiler the first time it installs.

## Limitations

- Stops at the first match; it does not list every appearance.
- Uses only the first face found in the reference photo.
- Shrinking frames to 25% can miss small or distant faces.
- Accuracy drops with poor lighting, side angles and low-resolution video.
- A tolerance of 0.6 is the library default and allows some false matches. Lower it (for example 0.5) for stricter matching.

## Possible improvements

- Return all matches with their timestamps.
- Support several faces in the reference photo.
- Show the matched frame with a box around the face.
- Make the tolerance and frame-skip adjustable in the app.
