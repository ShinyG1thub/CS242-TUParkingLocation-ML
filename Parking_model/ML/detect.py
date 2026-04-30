from ultralytics import YOLO
import cv2
import numpy as np
import json

# =========================
# 1) path
# =========================
img_path = "IMG_8314.PNG"   # เปลี่ยนชื่อได้ถ้าใช้รูปอื่น
slot_path = "slots.json"

# =========================
# 2) load model
# =========================
# ให้ใช้เหมือน Colab
model = YOLO("yolov8n.pt")   # ถ้าช้า ค่อยเปลี่ยนเป็น yolov8n.pt

# =========================
# 3) load image
# =========================
img = cv2.imread(img_path)

if img is None:
    raise Exception("❌ โหลดรูปไม่ได้")

img = cv2.resize(img, (640, 480))  # ต้องตรงกับตอนวาด polygon
output = img.copy()

# =========================
# 4) load slots
# =========================
with open(slot_path, "r", encoding="utf-8") as f:
    slots = json.load(f)

print("จำนวนช่อง:", len(slots))

# =========================
# 5) detect car
# =========================
results = model(img, conf=0.25, verbose=False)

car_boxes = []

for box in results[0].boxes:
    cls = int(box.cls[0])
    name = model.names[cls]

    if name in ["car", "truck", "bus"] and float(box.conf[0]) > 0.4:
        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()

        # ตัดกล่องเล็กเกินไป
        if (bx2 - bx1) < 30 or (by2 - by1) < 30:
            continue

        car_boxes.append([bx1, by1, bx2, by2])

print("จำนวนรถ:", len(car_boxes))

# =========================
# 6) utils
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


# =========================
# 7) check occupied
# =========================
def is_occupied(poly, car_boxes):
    # bounding box ของ polygon
    px = [p[0] for p in poly]
    py = [p[1] for p in poly]
    poly_box = [min(px), min(py), max(px), max(py)]

    for car in car_boxes:
        bx1, by1, bx2, by2 = car

        # method 1: center point
        cx = int((bx1 + bx2) / 2)
        cy = int((by1 + by2) / 2)

        if point_in_polygon(poly, (cx, cy)):
            return True

        # method 2: IoU กันพลาด
        iou = compute_iou(poly_box, car)
        if iou > 0.1:
            return True

    return False


# =========================
# 8) draw vehicle boxes
# =========================
for car in car_boxes:
    bx1, by1, bx2, by2 = map(int, car)
    cv2.rectangle(output, (bx1, by1), (bx2, by2), (255, 200, 0), 1)


# =========================
# 9) draw result
# =========================
empty_count = 0
full_count = 0

for poly in slots:
    occupied = is_occupied(poly, car_boxes)

    if occupied:
        full_count += 1
    else:
        empty_count += 1

    color = (0, 0, 255) if occupied else (0, 255, 0)
    text = "FULL" if occupied else "EMPTY"

    pts = np.array(poly, np.int32)
    cv2.polylines(output, [pts], True, color, 2)

    x, y = pts[0]
    cv2.putText(
        output,
        text,
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

# =========================
# 10) show result
# =========================
cv2.putText(
    output,
    f"Empty: {empty_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 0),
    2
)

cv2.putText(
    output,
    f"Full: {full_count}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.putText(
    output,
    f"Cars detected: {len(car_boxes)}",
    (20, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 200, 0),
    2
)

cv2.imshow("Parking Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()