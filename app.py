import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 YOLOv8 Object Detection")

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

source = st.sidebar.selectbox(
    "Choose Input Source",
    ["Camera", "Image Upload", "Video Upload"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

# -----------------------------
# Detection Function
# -----------------------------
def detect_objects(image):
    results = model(image)

    annotated_image = image.copy()

    for result in results:
        boxes = result.boxes

        if boxes is not None:
            for box in boxes:

                conf = float(box.conf[0])

                if conf < confidence:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_image,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10 if y1 > 20 else y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return annotated_image, results

# -----------------------------
# CAMERA INPUT
# -----------------------------
if source == "Camera":

    st.subheader("Capture Image")

    camera_image = st.camera_input("Take a Picture")

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting objects..."):
            output_image, results = detect_objects(image_np)

        st.image(
            output_image,
            caption="Detection Result",
            use_container_width=True
        )

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
elif source == "Image Upload":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting objects..."):
            output_image, results = detect_objects(image_np)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image_np,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                output_image,
                caption="Detected Objects",
                use_container_width=True
            )

        detected = []

        for result in results:
            for box in result.boxes:

                conf = float(box.conf[0])

                if conf >= confidence:
                    cls = int(box.cls[0])
                    detected.append(
                        f"{model.names[cls]} ({conf:.2f})"
                    )

        st.subheader("Detected Objects")

        if detected:
            st.write(", ".join(detected))
        else:
            st.warning("No objects detected.")

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
elif source == "Video Upload":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(uploaded_video.read())
        video_path = tfile.name

        stframe = st.empty()

        cap = cv2.VideoCapture(video_path)

        try:

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                results = model(frame)

                for result in results:

                    for box in result.boxes:

                        conf = float(box.conf[0])

                        if conf < confidence:
                            continue

                        cls = int(box.cls[0])
                        label = model.names[cls]

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                stframe.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

            st.success("Video Processing Completed!")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            cap.release()

            if os.path.exists(video_path):
                os.remove(video_path)