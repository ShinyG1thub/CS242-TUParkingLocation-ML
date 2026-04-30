from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import json
import requests
import os
from ultralytics import YOLO

app = FastAPI()

# =========================
# โหลด YOLO จาก S3
# =========================
YOLO_URL = "https://cs242-tuparkinglocation-ml.s3.us-east-1.amazonaws.com/yolov8x.pt"
YOLO_PATH = "yolov8x.pt"

def download_model():
    if not os.path.exists(YOLO_PATH):
        print("Downloading YOLO...")
        r = requests.get(YOLO_URL)
        with open(YOLO_PATH, "wb") as f:
            f.write(r.content)
        print("Done")

download_model()

model = YOLO(YOLO_PATH)

# =========================
# โหลด slots
# =========================
with open("slots.json", "r") as f:
    slots = json.load(f)

# =========================
# utils (เอามาจาก detect.py)
# =========================
def point_in_polygon(poly, point):
    return cv2.pointPolygonTest(
        np.array(poly, np.int32), point, False
    ) >= 0


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(x1, bx1)
    inter_y1 = max(y1, by1)
    inter_x2 = min(x2, bx2)
    inter_y2 = min(y2, by2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (bx2 - bx1) * (by2 - by1)

    return inter_area / (box1_area + box2_area - inter_area)


def is_occupied(poly, car_boxes):
    px = [p[0] for p in poly]
    py = [p[1] for p in poly]
    poly_box = [min(px), min(py), max(px), max(py)]

    for car in car_boxes:
        bx1, by1, bx2, by2 = car

        cx = int((bx1 + bx2) / 2)
        cy = int((by1 + by2) / 2)

        if point_in_polygon(poly, (cx, cy)):
            return True

        if compute_iou(poly_box, car) > 0.1:
            return True

    return False

# =========================
# API
# =========================
@app.get("/")
def root():
    return {"status": "ML service running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (640, 480))

    results = model(img, conf=0.25, verbose=False)

    car_boxes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["car", "truck", "bus"] and float(box.conf[0]) > 0.4:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()

            if (bx2 - bx1) < 30 or (by2 - by1) < 30:
                continue

            car_boxes.append([bx1, by1, bx2, by2])

    empty_count = 0
    result = []

    for poly in slots:
        occupied = is_occupied(poly, car_boxes)

        if not occupied:
            empty_count += 1

        result.append({
            "occupied": occupied
        })

    return {
        "empty": empty_count,
        "total_slots": len(slots),
        "cars_detected": len(car_boxes),
        "slots": result
    }