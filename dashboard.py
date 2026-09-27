import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(page_title="Smart Traffic AI - Chennai", layout="wide")
st.title("🚦 Smart Traffic AI Dashboard - Chennai")
st.markdown("Real-time Vehicle Counting using YOLOv8")

model = YOLO('yolov8n.pt')

uploaded = st.file_uploader("Upload traffic video", type=['mp4','avi'])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    count_chart = st.empty()
    counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[2,3,5,7])
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        vehicle_count = len(results[0].boxes)
        counts.append(vehicle_count)

        stframe.image(annotated, caption=f"Vehicles: {vehicle_count}")

        if len(counts) > 1:
            st.line_chart(counts)

    st.success(f"Total frames processed: {len(counts)} | Max Traffic: {max(counts)} vehicles")
else:
    st.info("traffic.mp4 upload pannu da")