import streamlit as st
import cv2
import face_recognition
import pickle
import os
from datetime import datetime
import pandas as pd

# ---- CONFIG ----
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.title("🎯 Face Recognition Attendance System")

# ---- Load encodings ----
ENCODINGS_FILE = "encodings.pickle"

@st.cache_resource
def load_encodings():
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)

data = load_encodings()

# ---- Time Range Inputs ----
col1, col2 = st.columns(2)
with col1:
    start_time = st.time_input("Start Time", value=datetime.strptime("09:00", "%H:%M").time())
with col2:
    end_time = st.time_input("End Time", value=datetime.strptime("17:00", "%H:%M").time())

# ---- Create New Unique Filename for Current Time Range ----
today_str = datetime.now().strftime("%d-%m-%Y")
time_range_str = f"{start_time.strftime('%H-%M')}_{end_time.strftime('%H-%M')}"
FILENAME = f"attendance_{today_str}_{time_range_str}.csv"

# ---- Webcam Attendance Button ----
start_button = st.button("📸 Start Attendance via Webcam")

if start_button:
    # Create a fresh CSV file
    with open(FILENAME, "w") as f:
        f.write("Name,Timestamp\n")

    st.warning("Press 'Q' on the webcam window to stop attendance.")
    video_stream = cv2.VideoCapture(0)

    recorded_names = set()

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        for encoding, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    matched_name = data["names"][i]
                    counts[matched_name] = counts.get(matched_name, 0) + 1
                name = max(counts, key=counts.get)

            timestamp = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

            if name != "Unknown" and name not in recorded_names:
                with open(FILENAME, "a") as f:
                    f.write(f"{name},{timestamp}\n")
                recorded_names.add(name)
                print(f"[LOG] {name} marked present at {timestamp}")

            # Draw box and label
            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()
    st.success("✅ Attendance session completed!")

# ---- Show Attendance Log ----
if os.path.exists(FILENAME):
    st.subheader("📋 Attendance Log")
    df = pd.read_csv(FILENAME)
    st.dataframe(df)

    # ---- Download Button ----
    st.download_button(
        label="⬇️ Download This Session's Attendance CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=FILENAME,
        mime='text/csv'
    )
