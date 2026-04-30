"""Data preparation utilities for ML operations."""

from typing import List, Tuple, Dict, Any
from app.models.parking import ParkingArea, ParkingSlot


class DataPreparer:
    """Utilities for preparing data for ML models from database."""

    @staticmethod
    def get_parking_area_features(parking_area_id: int) -> Dict[str, Any]:
        """
        Extract features for a parking area for model input.
        
        Returns:
            Dictionary with parking area features
        """
        area = ParkingArea.query.get(parking_area_id)
        if not area:
            return {}
        
        # Get current parking status
        occupied_slots = len([s for s in area.slots if s.status == 'occupied'])
        available_slots = len([s for s in area.slots if s.status == 'available'])
        
        features = {
            'area_id': area.id,
            'name': area.name,
            'total_slots': area.total_slots,
            'available_slots': available_slots,
            'occupied_slots': occupied_slots,
            'occupancy_rate': occupied_slots / area.total_slots if area.total_slots > 0 else 0,
            'latitude': area.latitude,
            'longitude': area.longitude
        }
        return features

    @staticmethod
    def get_all_areas_features() -> List[Dict[str, Any]]:
        """Get features for all parking areas."""
        areas = ParkingArea.query.all()
        return [DataPreparer.get_parking_area_features(area.id) for area in areas]

    @staticmethod
    def prepare_training_data(
        parking_areas: List[int] = None,
        look_back_hours: int = 24
    ) -> Tuple[List[Dict], List[str]]:
        """
        Prepare historical data for model training.
        
        Args:
            parking_areas: List of parking area IDs (None = all areas)
            look_back_hours: Historical data window in hours
            
        Returns:
            Tuple of (feature_list, label_list)
        """
        if not parking_areas:
            parking_areas = [a.id for a in ParkingArea.query.all()]
        
        features_list = []
        labels_list = []
        
        for area_id in parking_areas:
            features = DataPreparer.get_parking_area_features(area_id)
            if features:
                features_list.append(features)
                # Label based on occupancy rate (example logic)
                occupancy = features.get('occupancy_rate', 0)
                if occupancy > 0.8:
                    label = 'likely_full'
                elif occupancy > 0.5:
                    label = 'moderate'
                else:
                    label = 'available'
                labels_list.append(label)
        
        return features_list, labels_list

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature names used in models."""
        return [
            'area_id',
            'name',
            'total_slots',
            'available_slots',
            'occupied_slots',
            'occupancy_rate',
            'latitude',
            'longitude'
        ]

    @staticmethod
    def normalize_features(features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize features for ML model input.
        
        Simple min-max normalization for numerical features.
        """
        if not features_list:
            return []
        
        # Find min/max for numeric features
        numeric_features = ['occupancy_rate']
        min_max = {}
        
        for feat in numeric_features:
            values = [f.get(feat, 0) for f in features_list if feat in f]
            if values:
                min_max[feat] = {
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values) or 1
                }
        
        # Normalize
        normalized = []
        for features in features_list:
            norm_features = dict(features)
            for feat, stats in min_max.items():
                if feat in norm_features:
                    norm_features[feat] = (
                        (features[feat] - stats['min']) / stats['range']
                    )
            normalized.append(norm_features)
        
        return normalized
