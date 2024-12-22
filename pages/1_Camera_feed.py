import streamlit as st
import cv2
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO('fire-models/fire_l.pt')

# Detection settings
st.title("Fire and Smoke Detection System")
st.sidebar.header("Detection Settings")

# Load model
model = load_model()

# Add confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_sent_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Resize the frame to reduce processing time while maintaining aspect ratio
        img = cv2.resize(img, (640, 480))
        
        # Perform detection
        results = model.predict(
            source=img,
            conf=confidence_threshold,
            device='cpu'  # Use 'cuda' if you have GPU support
        )

        # Get the plotted frame with detections
        annotated_frame = results[0].plot()
        
        # Convert back to BGR format for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Add detection info
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[class_id]
                
                # Log detection
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                st.sidebar.write(f"Detected {class_name} with {conf:.2f} confidence at {current_time}")

        return av.VideoFrame.from_ndarray(annotated_frame, format='bgr24')

# Start the webrtc streamer
webrtc_streamer(
    key="fire-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={
        "video": {"width": 640, "height": 480},
        "audio": False
    },
    async_processing=True
)
