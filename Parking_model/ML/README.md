# ML Module - TU Parking Location Prediction

## Overview
The ML module manages machine learning models for predicting parking availability at TU Parking locations. It integrates with the database to store model artifacts, predictions, and training history.

## Project Structure

```
ml/
├── __init__.py
├── models/                    # Trained model artifacts storage
│   └── __init__.py
├── data/                      # Training and evaluation datasets
│   └── __init__.py
├── services/                  # ML services and business logic
│   ├── __init__.py
│   └── parking_prediction_service.py   # Prediction service
└── utils/                     # Utility functions
    ├── __init__.py
    └── data_preparer.py       # Data preparation utilities
```

## Database Models

Three new models have been added to `app/models/ml_models.py`:

### 1. **MLModel**
Stores metadata about trained ML models.

**Fields:**
- `id`: Primary key
- `name`: Unique model name
- `model_type`: Type of model (e.g., 'RandomForest', 'Neural Network')
- `version`: Version string (e.g., '1.0.0')
- `file_path`: Path to saved model artifact
- `accuracy`, `precision`, `recall`, `f1_score`: Performance metrics
- `is_active`: Flag for current production model
- `created_at`, `updated_at`: Timestamps

**Relationships:**
- One-to-many with `Prediction`
- One-to-many with `TrainingHistory`

### 2. **Prediction**
Stores individual prediction results linked to parking areas.

**Fields:**
- `id`: Primary key
- `model_id`: Foreign key to MLModel
- `parking_area_id`: Foreign key to ParkingArea
- `prediction_value`: Prediction class (e.g., 'likely_full', 'available', 'moderate')
- `confidence_score`: Confidence level (0.0 to 1.0)
- `predicted_available_slots`: Predicted number of available slots
- `input_features`: JSON string of input features used
- `is_accurate`: Validation flag (set after actual data is known)
- `created_at`: Timestamp

### 3. **TrainingHistory**
Records training sessions and performance metrics.

**Fields:**
- `id`: Primary key
- `model_id`: Foreign key to MLModel
- `training_start_time`, `training_end_time`: Training duration
- `training_duration_seconds`: Total training time
- `training_samples_count`: Number of samples used
- `training_accuracy`, `validation_accuracy`: Accuracy metrics
- `training_loss`, `validation_loss`: Loss metrics
- `status`: Training status ('in_progress', 'completed', 'failed')
- `error_message`: Error details if training failed
- `notes`: Additional hyperparameters/notes

## Core Classes

### MLManager (`app/services/ml_manager.py`)
Manages all ML-related database operations.

**Key Methods:**

**Model Management:**
- `add_ml_model()`: Add a new trained model
- `get_all_ml_models()`: Retrieve all models
- `get_ml_model_by_id()`: Get specific model
- `get_active_model()`: Get current production model
- `set_active_model()`: Set production model
- `update_model_metrics()`: Update model performance
- `delete_ml_model()`: Delete model and related data

**Prediction Management:**
- `add_prediction()`: Store a prediction result
- `get_predictions_by_area()`: Get recent predictions for an area
- `get_predictions_by_model()`: Get all predictions from a model
- `get_latest_prediction_for_area()`: Get most recent prediction
- `mark_prediction_accuracy()`: Validate prediction accuracy

**Training Management:**
- `start_training_session()`: Create new training record
- `end_training_session()`: Complete training with metrics
- `get_training_history()`: Get all training records for a model
- `fail_training_session()`: Mark training as failed

### DataPreparer (`ml/utils/data_preparer.py`)
Prepares data from database for ML operations.

**Key Methods:**
- `get_parking_area_features()`: Extract features for one area
- `get_all_areas_features()`: Extract features for all areas
- `prepare_training_data()`: Prepare historical data for training
- `get_feature_names()`: List of feature names
- `normalize_features()`: Min-max normalization

**Features Extracted:**
- Area ID and name
- Total slots
- Available/occupied slots
- Occupancy rate (0-1)
- Latitude/longitude

### ParkingPredictionService (`ml/services/parking_prediction_service.py`)
High-level service for making predictions.

**Key Methods:**
- `make_prediction()`: Predict for one parking area
- `predict_all_areas()`: Predict for all areas
- `get_prediction_history()`: Get recent predictions
- `get_active_model_info()`: Get active model details

## Usage Examples

### 1. Add a New Trained Model
```python
from app.services.ml_manager import MLManager

ml_manager = MLManager()

# Add a new model
model = ml_manager.add_ml_model(
    name='RandomForest_v1',
    model_type='RandomForest',
    version='1.0.0',
    file_path='ml/models/rf_model_v1.pkl',
    description='Random Forest model trained on 6 months of data',
    accuracy=0.92,
    precision=0.89,
    recall=0.94,
    f1_score=0.915
)

# Set it as active
ml_manager.set_active_model(model.id)
```

### 2. Make Predictions
```python
from ml.services import ParkingPredictionService

service = ParkingPredictionService()

# Predict for one area
result = service.make_prediction(parking_area_id=1)
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']})")

# Predict for all areas
all_predictions = service.predict_all_areas()
```

### 3. Track Training
```python
# Start training session
session = ml_manager.start_training_session(
    model_id=1,
    notes='Hyperparameters: max_depth=10, n_estimators=100'
)

# ... training code ...

# End training with metrics
ml_manager.end_training_session(
    session_id=session.id,
    training_samples_count=5000,
    training_accuracy=0.95,
    validation_accuracy=0.92,
    training_loss=0.15,
    validation_loss=0.18
)
```

### 4. Validate Predictions
```python
# After actual parking data is known, mark if prediction was correct
ml_manager.mark_prediction_accuracy(prediction_id=123, is_accurate=True)
```

## Integration with Flask Routes

To add prediction endpoints to your API, add to `app/routes/parking_routes.py`:

```python
from ml.services import ParkingPredictionService

prediction_service = ParkingPredictionService()

@parking_bp.route('/api/predict/<int:area_id>', methods=['GET'])
def predict_area(area_id):
    """Get prediction for a parking area."""
    result = prediction_service.make_prediction(area_id)
    return jsonify(result)

@parking_bp.route('/api/predict/all', methods=['GET'])
def predict_all():
    """Get predictions for all parking areas."""
    result = prediction_service.predict_all_areas()
    return jsonify(result)

@parking_bp.route('/api/predictions/history/<int:area_id>', methods=['GET'])
def get_prediction_history(area_id):
    """Get prediction history for an area."""
    limit = request.args.get('limit', 10, type=int)
    history = prediction_service.get_prediction_history(area_id, limit)
    return jsonify(history)
```

## Database Schema Updates

Run migrations to create the new tables:

```sql
-- MLModel table
CREATE TABLE ml_models (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    description TEXT
);

-- Prediction table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL FOREIGN KEY REFERENCES ml_models(id),
    parking_area_id INTEGER NOT NULL FOREIGN KEY REFERENCES parking_areas(id),
    prediction_value VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
    predicted_available_slots INTEGER,
    input_features TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_accurate BOOLEAN
);

-- TrainingHistory table
CREATE TABLE training_history (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL FOREIGN KEY REFERENCES ml_models(id),
    training_start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    training_end_time DATETIME,
    training_duration_seconds FLOAT,
    training_samples_count INTEGER,
    validation_split_ratio FLOAT DEFAULT 0.2,
    training_loss FLOAT,
    validation_loss FLOAT,
    training_accuracy FLOAT,
    validation_accuracy FLOAT,
    status VARCHAR(20) DEFAULT 'in_progress',
    error_message TEXT,
    notes TEXT
);
```

## Next Steps

1. **Model Training Pipeline**: Implement actual model training in `ml/services/parking_prediction_service.py`
2. **Model Persistence**: Save trained models to `ml/models/` directory
3. **Real Inference**: Replace mock prediction logic with actual model loading and inference
4. **API Integration**: Add prediction routes to Flask app
5. **Frontend Display**: Show predictions and confidence scores in the UI
6. **Validation**: Track prediction accuracy against actual parking data

## File Structure Summary

- `app/models/ml_models.py`: SQLAlchemy models for ML data
- `app/services/ml_manager.py`: Database operations manager
- `ml/services/parking_prediction_service.py`: High-level prediction service
- `ml/utils/data_preparer.py`: Data extraction and preparation utilities
- `ml/models/`: Directory for saved model artifacts
- `ml/data/`: Directory for training datasets

## Dependencies

Required packages (add to `requirements.txt`):
- SQLAlchemy (already in project)
- scikit-learn (for ML models)
- pandas (for data manipulation)
- numpy (for numerical operations)

Example additions to `requirements.txt`:
```
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
```
