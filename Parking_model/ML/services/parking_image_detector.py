"""Image-based parking slot detection service."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class ParkingImageDetector:
    """Run parking occupancy detection on an uploaded image."""

    BASE_DIR = Path(__file__).resolve().parents[1]
    TARGET_SIZE = (640, 480)
    CONFIDENCE_THRESHOLD = 0.4
    VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._slots: list[list[list[int]]] | None = None

    @property
    def model_path(self) -> Path:
        preferred = self.BASE_DIR / "yolov8x.pt"
        if preferred.exists():
            return preferred
        fallback = self.BASE_DIR / "parking_model.pt"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No ML model file found in ML directory")

    @property
    def slots_path(self) -> Path:
        path = self.BASE_DIR / "slots.json"
        if not path.exists():
            raise FileNotFoundError("slots.json was not found in ML directory")
        return path

    def get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(str(self.model_path))
        return self._model

    def get_slots(self) -> list[list[list[int]]]:
        if self._slots is None:
            self._slots = json.loads(self.slots_path.read_text(encoding="utf-8"))
        return self._slots

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file is not a valid image")
        return cv2.resize(image, ParkingImageDetector.TARGET_SIZE)

    @staticmethod
    def _point_in_polygon(poly: list[list[int]], point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(np.array(poly, np.int32), point, False) >= 0

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
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

    def _extract_vehicle_boxes(self, image: np.ndarray) -> list[list[float]]:
        model = self.get_model()
        results = model(image, conf=0.25, verbose=False)
        car_boxes: list[list[float]] = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            confidence = float(box.conf[0])

            if name not in self.VEHICLE_CLASSES or confidence < self.CONFIDENCE_THRESHOLD:
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
            if (bx2 - bx1) < 30 or (by2 - by1) < 30:
                continue

            car_boxes.append([bx1, by1, bx2, by2])

        return car_boxes

    def _is_occupied(self, poly: list[list[int]], car_boxes: list[list[float]]) -> bool:
        px = [point[0] for point in poly]
        py = [point[1] for point in poly]
        poly_box = [min(px), min(py), max(px), max(py)]

        for car in car_boxes:
            bx1, by1, bx2, by2 = car
            center_x = int((bx1 + bx2) / 2)
            center_y = int((by1 + by2) / 2)

            if self._point_in_polygon(poly, (center_x, center_y)):
                return True

            if self._compute_iou(poly_box, car) > 0.1:
                return True

        return False

    @staticmethod
    def _encode_preview(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode annotated image")
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        image = self._decode_image(image_bytes)
        output = image.copy()
        slots = self.get_slots()
        car_boxes = self._extract_vehicle_boxes(image)

        for car in car_boxes:
            bx1, by1, bx2, by2 = map(int, car)
            cv2.rectangle(output, (bx1, by1), (bx2, by2), (255, 200, 0), 1)

        slot_results: list[dict[str, Any]] = []
        empty_count = 0
        full_count = 0

        for index, poly in enumerate(slots, start=1):
            occupied = self._is_occupied(poly, car_boxes)
            status = "occupied" if occupied else "available"
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
                2,
            )

            slot_results.append(
                {
                    "slot_index": index,
                    "status": status,
                    "polygon": poly,
                }
            )

        cv2.putText(
            output,
            f"Empty: {empty_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            output,
            f"Full: {full_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            output,
            f"Cars detected: {len(car_boxes)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
        )

        return {
            "model_name": self.model_path.name,
            "total_slots_analyzed": len(slot_results),
            "available_slots": empty_count,
            "occupied_slots": full_count,
            "cars_detected": len(car_boxes),
            "slot_results": slot_results,
            "annotated_image": self._encode_preview(output),
        }


parking_image_detector = ParkingImageDetector()
