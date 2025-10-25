import cv2
import face_recognition
import streamlit as st
import tempfile

st.header("FrameTrace by Talha Butt")
st.divider()

# Upload target image
target_file = st.file_uploader("Please upload the image you want to find:", type=["jpg", "png", "webp"])
video_file = st.file_uploader("Please upload the video:", type=["mp4", "avi", "mov"])

if target_file and video_file:
    # Encode target face
    target = face_recognition.load_image_file(target_file)
    target_encs = face_recognition.face_encodings(target)
    if not target_encs:
        st.error("No face detected in target image.")
        st.stop()
    target_enc = target_encs[0]

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video = cv2.VideoCapture(tfile.name)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    found = False

    progress = st.progress(0, text="Scanning video...")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 5 != 0:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # rgb_frame = small_frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        encs = face_recognition.face_encodings(rgb_frame)
        if not encs:
            continue

        for enc in encs:
            res = face_recognition.compare_faces([target_enc], enc, tolerance=0.6)
            if res[0]:
                found = True
                timestamp = round(frame_num / fps, 2)
                st.success(f"Face found at {timestamp} seconds.")
                break

        progress.progress(min(frame_num / 1000, 1.0))  # basic progress indicator

        if found:
            break

    video.release()
    if not found:
        st.warning("No matching face found in the video.")
