# CS242-TUParkingLocation-ML

Machine Learning module for TU Parking Location, including image-based parking slot detection with YOLO.
resource: https://drive.google.com/drive/folders/1TVxRAQsIOhJNHmLgDyn_ioq2BS7dwxKI?usp=sharing

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ML
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

If you use the database prediction service, also install the backend dependencies from the main application, such as SQLAlchemy, pandas, and scikit-learn.

### 4. Add model weights

Large model files are ignored by Git, so place the model file manually in:

```text
Parking_model/ML/yolov8x.pt
```

or:

```text
Parking_model/ML/parking_model.pt
```

The image detector will use `yolov8x.pt` first. If it is not found, it will fall back to `parking_model.pt`.

### 5. Run the image detection demo

```bash
cd Parking_model/ML
python detect.py
```

The demo uses:

- `IMG_8314.PNG` as the input image
- `slots.json` as the parking slot polygon file
- `yolov8x.pt` or `parking_model.pt` as the YOLO model

## Project Files

- `Parking_model/ML/detect.py`: local image detection demo
- `Parking_model/ML/services/parking_image_detector.py`: reusable parking image detector service
- `Parking_model/ML/slots.json`: parking slot polygon definitions
- `Parking_model/ML/README.md`: detailed ML module documentation
