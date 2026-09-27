import cv2
from ultralytics import YOLO
import pygame
import time
import os
import csv
from datetime import datetime

# --- CONFIGURATION ---
AUDIO_FILE = "bapas rakh usse copy.mp3"
CONFIDENCE_LEVEL = 0.5
COOLDOWN_SECONDS = 11
FRAME_SKIP = 1

violation_count = 0
# ---------------------

if not os.path.exists(AUDIO_FILE):
    print(f"ERROR: '{AUDIO_FILE}' file nahi mili!")
    exit()

# Audio Load
pygame.mixer.init()
try:
    pygame.mixer.music.load(AUDIO_FILE)
except Exception as e:
    print(f"Audio Error: {e}")
    exit()

print("Loading Fast AI Model...")
model = YOLO("yolov8n.pt")
model.fuse()

cap = cv2.VideoCapture(0)
cv2.namedWindow("No Phone Zone", cv2.WINDOW_NORMAL)
cv2.resizeWindow("No Phone Zone", 900, 600)

# Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

last_alert_time = 0
frame_count = 0
phone_detected = False
boxes_to_draw = []

print("System Ready. Press 'q' to exit.")

# CSV file create
if not os.path.exists("violations.csv"):
    with open("violations.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Event"])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # AI detection every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:

        results = model(
            frame,
            stream=True,
            verbose=False,
            conf=CONFIDENCE_LEVEL,
            imgsz=256
        )

        phone_detected = False
        boxes_to_draw = []

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name == "cell phone":

                    phone_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes_to_draw.append((x1, y1, x2, y2))

    # Draw Detection
    if phone_detected:

        for (x1, y1, x2, y2) in boxes_to_draw:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "PHONE RAKH NEECHE!",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        current_time = time.time()

        if current_time - last_alert_time > COOLDOWN_SECONDS:

            violation_count += 1

            print(f">>> ALERT #{violation_count}")

            # Screenshot Save
            filename = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            # CSV Log Save
            with open("violations.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"Phone Detected ({violation_count})"
                ])

            # Play Audio
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

            last_alert_time = current_time

    # Show Counter
    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("No Phone Zone", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
 #py -3.12 app.py