from pathlib import Path
import json
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_NAME = "IMG_8314.PNG"
OUTPUT_PATH = BASE_DIR / "slots.json"
TARGET_SIZE = (640, 480)

slots = []
current = []


def resolve_image_path() -> Path:
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        return candidate if candidate.is_absolute() else BASE_DIR / candidate

    default_image = BASE_DIR / DEFAULT_IMAGE_NAME
    if default_image.exists():
        return default_image

    png_files = sorted(BASE_DIR.glob("*.PNG"))
    if png_files:
        return png_files[0]

    raise FileNotFoundError("No PNG image was found next to polygon.py")


def click(event, x, y, flags, param):
    global current

    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
        print(f"Point: {x},{y}")


def main() -> None:
    image_path = resolve_image_path()
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"โหลดรูปไม่ได้: {image_path}")
        raise SystemExit(1)

    img = cv2.resize(img, TARGET_SIZE)

    cv2.namedWindow("Draw Slots")
    cv2.setMouseCallback("Draw Slots", click)

    while True:
        temp = img.copy()

        for poly in slots:
            pts = np.array(poly, np.int32)
            cv2.polylines(temp, [pts], True, (0, 255, 0), 2)

        if current:
            pts = np.array(current, np.int32)
            cv2.polylines(temp, [pts], False, (255, 0, 0), 2)

            for point in current:
                cv2.circle(temp, point, 4, (0, 0, 255), -1)

        cv2.putText(
            temp,
            "n: new slot | z: undo point | u: undo slot | s: save | esc: exit",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Draw Slots", temp)
        key = cv2.waitKey(1)

        if key == ord("n"):
            if len(current) >= 3:
                slots.append(current.copy())
                current.clear()
                print(f"Saved slot count: {len(slots)}")

        elif key == ord("z"):
            if current:
                current.pop()
                print("Undo point")

        elif key == ord("u"):
            if slots:
                slots.pop()
                print("Undo slot")

        elif key == ord("s"):
            with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
                json.dump(slots, file, ensure_ascii=False)
            print(f"Saved polygons to {OUTPUT_PATH}")

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
