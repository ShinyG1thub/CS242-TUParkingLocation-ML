"""ML Services module initialization."""

__all__ = [
    "ParkingPredictionService",
    "ParkingImageDetector",
    "parking_image_detector",
]


def __getattr__(name: str):
    if name == "ParkingPredictionService":
        from .parking_prediction_service import ParkingPredictionService

        return ParkingPredictionService
    if name in {"ParkingImageDetector", "parking_image_detector"}:
        from .parking_image_detector import ParkingImageDetector, parking_image_detector

        return {
            "ParkingImageDetector": ParkingImageDetector,
            "parking_image_detector": parking_image_detector,
        }[name]
    raise AttributeError(name)
