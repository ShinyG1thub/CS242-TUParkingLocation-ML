"""ML Service for model training and inference."""

from typing import Dict, Any, List
import json
from app.services.ml_manager import MLManager
from ml.utils.data_preparer import DataPreparer


class ParkingPredictionService:
    """Service for parking availability prediction using ML models."""

    def __init__(self):
        self.ml_manager = MLManager()
        self.data_preparer = DataPreparer()

    def make_prediction(self, parking_area_id: int) -> Dict[str, Any]:
        """
        Make a prediction for a specific parking area using the active model.
        
        Args:
            parking_area_id: ID of the parking area to predict for
            
        Returns:
            Prediction result with confidence score
        """
        # Get active model
        active_model = self.ml_manager.get_active_model()
        if not active_model:
            return {
                'success': False,
                'error': 'No active ML model available'
            }

        # Get features for the area
        features = self.data_preparer.get_parking_area_features(parking_area_id)
        if not features:
            return {
                'success': False,
                'error': f'Parking area {parking_area_id} not found'
            }

        # Make prediction (placeholder - replace with actual model inference)
        prediction_result = self._predict_with_model(features, active_model)

        # Store prediction in database
        stored_prediction = self.ml_manager.add_prediction(
            model_id=active_model['id'],
            parking_area_id=parking_area_id,
            prediction_value=prediction_result['prediction'],
            confidence_score=prediction_result['confidence'],
            predicted_available_slots=prediction_result.get('predicted_slots'),
            input_features=features
        )

        return {
            'success': True,
            'prediction_id': stored_prediction.id,
            'parking_area_id': parking_area_id,
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'model_id': active_model['id'],
            'model_name': active_model['name']
        }

    def predict_all_areas(self) -> Dict[str, Any]:
        """Make predictions for all parking areas."""
        active_model = self.ml_manager.get_active_model()
        if not active_model:
            return {
                'success': False,
                'error': 'No active ML model available'
            }

        # Get features for all areas
        all_features = self.data_preparer.get_all_areas_features()
        
        predictions = []
        for features in all_features:
            prediction_result = self._predict_with_model(features, active_model)
            
            # Store each prediction
            stored_pred = self.ml_manager.add_prediction(
                model_id=active_model['id'],
                parking_area_id=features['area_id'],
                prediction_value=prediction_result['prediction'],
                confidence_score=prediction_result['confidence'],
                predicted_available_slots=prediction_result.get('predicted_slots'),
                input_features=features
            )
            
            predictions.append({
                'parking_area_id': features['area_id'],
                'area_name': features['name'],
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence']
            })

        return {
            'success': True,
            'model_name': active_model['name'],
            'total_predictions': len(predictions),
            'predictions': predictions
        }

    def get_prediction_history(self, parking_area_id: int, limit: int = 10) -> List[Dict]:
        """Get recent predictions for a parking area."""
        return self.ml_manager.get_predictions_by_area(parking_area_id, limit)

    def get_active_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model."""
        model = self.ml_manager.get_active_model()
        if not model:
            return {'model': None, 'message': 'No active model set'}
        return {'model': model}

    @staticmethod
    def _predict_with_model(features: Dict, model_info: Dict) -> Dict[str, Any]:
        """
        Internal method to make prediction with model.
        
        TODO: Replace with actual model loading and inference
        """
        occupancy_rate = features.get('occupancy_rate', 0)
        
        # Simple rule-based logic (replace with actual model prediction)
        if occupancy_rate > 0.8:
            prediction = 'likely_full'
            confidence = 0.85
        elif occupancy_rate > 0.5:
            prediction = 'moderate'
            confidence = 0.75
        else:
            prediction = 'available'
            confidence = 0.80

        return {
            'prediction': prediction,
            'confidence': confidence,
            'predicted_slots': max(0, int(features.get('available_slots', 0)))
        }
