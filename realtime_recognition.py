import face_recognition
import cv2
import pickle
import numpy as np

# Load known encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Start video stream
video_capture = cv2.VideoCapture(0)

print("[INFO] Starting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for speed (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    recognized_names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(data["encodings"], encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]

        recognized_names.append(name)

    # Draw boxes + names
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # Scale back up (we used 1/4 size)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
