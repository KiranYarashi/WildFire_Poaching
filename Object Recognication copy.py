# import cv2
# import matplotlib.pyplot as plt
# from twilio.rest import Client
# import time

# # Twilio credentials
# import streamlit as st
# # Access secrets securely
# account_sid = st.secrets["twilio"]["account_sid"]
# auth_token = st.secrets["twilio"]["auth_token"] 
# twilio_number = st.secrets["twilio"]["twilio_number"]
# target_number = st.secrets["twilio"]["target_number"]# Ensure the target number includes the country code

# client = Client(account_sid, auth_token)

# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'

# model = cv2.dnn_DetectionModel(frozen_model, config_file)

# classLabels = []
# file_name = 'Label.txt'
# with open(file_name, 'rt') as fpt:
#     classLabels = fpt.read().rstrip('\n').split('\n')

# print(classLabels)
# print(len(classLabels))

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot open video")

# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN

# last_sent_time = 0
# send_interval = 5 * 60  # 5 minutes in seconds

# animal_classes = [16, 17, 18, 19, 20, 21, 22, 23, 24]  # Example class IDs for animals

# while True:
#     ret, frame = cap.read()
#     ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

#     if len(ClassIndex) != 0:
#         for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
#             if ClassInd == 1:  # Class ID for person
#                 cv2.rectangle(frame, boxes, (255, 0, 0), 2)
#                 cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0), 3)

#                 current_time = time.time()
#                 if current_time - last_sent_time > send_interval:
#                     # Send SMS
#                     message = client.messages.create(
#                         body="Person detected!",
#                         from_=twilio_number,
#                         to=target_number
#                     )
#                     print(f"SMS sent: {message.sid}")
#                     last_sent_time = current_time
#             elif ClassInd in animal_classes:  # Check if the detected object is an animal
#                 cv2.rectangle(frame, boxes, (0, 255, 0), 2)
#                 cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (255, 0, 0), 3)

#                 current_time = time.time()
#                 if current_time - last_sent_time > send_interval:
#                     # Send SMS
#                     message = client.messages.create(
#                         body=f"Animal  detected!",
#                         from_=twilio_number,
#                         to=target_number
#                     )
#                     print(f"SMS sent: {message.sid}")
#                     last_sent_time = current_time

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import av
import matplotlib.pyplot as plt
from twilio.rest import Client
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Access secrets securely
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
twilio_number = st.secrets["twilio"]["twilio_number"]
target_number = st.secrets["twilio"]["target_number"]  # Ensure the target number includes the country code

client = Client(account_sid, auth_token)

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Label.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

send_interval = 5 * 60  # 5 minutes in seconds
animal_classes = [16, 17, 18, 19, 20, 21, 22, 23, 24]  # Example class IDs for animals

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_sent_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd == 1:  # Class ID for person
                    cv2.rectangle(img, boxes, (255, 0, 0), 2)
                    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                    current_time = time.time()
                    if current_time - self.last_sent_time > send_interval:
                        # Send SMS
                        message = client.messages.create(
                            body="Person detected!",
                            from_=twilio_number,
                            to=target_number
                        )
                        print(f"SMS sent: {message.sid}")
                        self.last_sent_time = current_time
                elif ClassInd in animal_classes:  # Check if the detected object is an animal
                    cv2.rectangle(img, boxes, (0, 255, 0), 2)
                    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                    current_time = time.time()
                    if current_time - self.last_sent_time > send_interval:
                        # Send SMS
                        message = client.messages.create(
                            body="Animal detected!",
                            from_=twilio_number,
                            to=target_number
                        )
                        print(f"SMS sent: {message.sid}")
                        self.last_sent_time = current_time

        return img

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)