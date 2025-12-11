"""
Carbon Emissions Prediction API
Deployed on Cloud Run with BigQuery Logging
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
from google.cloud import bigquery
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# ENABLE CORS FOR ALL ROUTES
# ============================================================================
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# ============================================================================
# LOAD MODEL COMPONENTS
# ============================================================================
print("\n" + "="*80)
print("🚀 LOADING CARBON EMISSIONS MODEL")
print("="*80)

try:
    model = joblib.load('carbon_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print("✅ Model loaded successfully!")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ============================================================================
# INITIALIZE BIGQUERY
# ============================================================================
try:
    bigquery_client = bigquery.Client()
    PROJECT_ID = bigquery_client.project
except Exception as e:
    logger.warning(f"BigQuery not available: {e}")
    bigquery_client = None
    PROJECT_ID = None

# ============================================================================
# HELPER: LOG TO BIGQUERY
# ============================================================================
def log_prediction_to_bigquery(input_data, prediction, confidence):
    """Log prediction to BigQuery"""
    
    if not bigquery_client:
        return False
    
    try:
        table_id = f"{PROJECT_ID}.carbonsense_data.predictions_log"
        
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "electricity_kwh": float(input_data['electricity_kwh']),
            "gas_therms": float(input_data['gas_therms']),
            "distance_km": float(input_data['distance_km']),
            "vehicle_type": str(input_data['vehicle_type']),
            "shopping_carbon_kg": float(input_data['shopping_carbon_kg']),
            "predicted_carbon_kg": float(prediction),
            "confidence": str(confidence),
            "model_version": "v1"
        }
        
        errors = bigquery_client.insert_rows_json(table_id, [row])
        
        if not errors:
            logger.info("✅ Prediction logged to BigQuery")
            return True
        else:
            logger.warning(f"BigQuery insert errors: {errors}")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to log to BigQuery: {e}")
        return False

# ============================================================================
# ENDPOINT 1: HEALTH CHECK
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'carbon-emissions-predictor',
        'version': 'v1'
    }), 200

# ============================================================================
# ENDPOINT 2: MODEL INFO
# ============================================================================
@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model_name': 'Carbon Emissions Predictor',
        'version': 'v1',
        'model_type': metrics['model_type'],
        'training_samples': metrics['total_training_samples'],
        'performance': metrics['metrics'],
        'features': {
            'electricity_kwh': 'Monthly electricity (kWh)',
            'gas_therms': 'Monthly gas (therms)',
            'distance_km': 'Monthly distance (km)',
            'vehicle_type': 'Vehicle type',
            'shopping_carbon_kg': 'Shopping carbon (kg)'
        },
        'valid_vehicle_types': list(label_encoder.classes_),
        'feature_importance': metrics['feature_importance']
    }), 200

# ============================================================================
# ENDPOINT 3: PREDICT (MAIN)
# ============================================================================
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Make a prediction
    
    Input JSON:
    {
        "electricity_kwh": 450.5,
        "gas_therms": 30.2,
        "distance_km": 500.0,
        "vehicle_type": "sedan",
        "shopping_carbon_kg": 100.5
    }
    """
    
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        request_json = request.get_json()
        
        if not request_json:
            return jsonify({'error': 'No JSON data'}), 400
        
        # Extract data
        electricity = float(request_json['electricity_kwh'])
        gas = float(request_json['gas_therms'])
        distance = float(request_json['distance_km'])
        vehicle_type = str(request_json['vehicle_type']).lower()
        shopping = float(request_json['shopping_carbon_kg'])
        
        # Validate vehicle type
        valid_vehicles = list(label_encoder.classes_)
        if vehicle_type not in valid_vehicles:
            return jsonify({
                'error': f'Invalid vehicle_type: {vehicle_type}',
                'valid_options': valid_vehicles
            }), 400
        
        # Encode and predict
        vehicle_encoded = label_encoder.transform([vehicle_type])[0]
        features = np.array([[electricity, gas, distance, vehicle_encoded, shopping]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Determine confidence
        if 500 < prediction < 800:
            confidence = "high"
        else:
            confidence = "medium"
        
        # Log to BigQuery
        input_data = {
            'electricity_kwh': electricity,
            'gas_therms': gas,
            'distance_km': distance,
            'vehicle_type': vehicle_type,
            'shopping_carbon_kg': shopping
        }
        log_prediction_to_bigquery(input_data, prediction, confidence)
        
        response = {
            'prediction': {
                'predicted_carbon_kg': round(float(prediction), 2),
                'confidence': confidence,
                'model_version': 'v1'
            },
            'input': input_data,
            'model_performance': {
                'test_r2': metrics['metrics']['test_r2'],
                'test_mae': metrics['metrics']['test_mae']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# ============================================================================
# ENDPOINT 4: BATCH PREDICTIONS
# ============================================================================
@app.route('/predict-batch', methods=['POST', 'OPTIONS'])
def predict_batch():
    """Make multiple predictions"""
    
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        request_json = request.get_json()
        instances = request_json.get('instances', [])
        
        if not instances:
            return jsonify({'error': 'No instances provided'}), 400
        
        predictions = []
        
        for instance in instances:
            try:
                electricity = float(instance['electricity_kwh'])
                gas = float(instance['gas_therms'])
                distance = float(instance['distance_km'])
                vehicle_type = str(instance['vehicle_type']).lower()
                shopping = float(instance['shopping_carbon_kg'])
                
                vehicle_encoded = label_encoder.transform([vehicle_type])[0]
                features = np.array([[electricity, gas, distance, vehicle_encoded, shopping]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                
                input_data = {
                    'electricity_kwh': electricity,
                    'gas_therms': gas,
                    'distance_km': distance,
                    'vehicle_type': vehicle_type,
                    'shopping_carbon_kg': shopping
                }
                log_prediction_to_bigquery(input_data, prediction, "batch")
                
                predictions.append({
                    'predicted_carbon_kg': round(float(prediction), 2),
                    'input': instance
                })
                
            except Exception as e:
                predictions.append({'error': str(e), 'input': instance})
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.route('/', methods=['GET'])
def root():
    """API documentation"""
    return jsonify({
        'service': 'Carbon Emissions Prediction API',
        'version': 'v1',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'GET /info': 'Model information',
            'POST /predict': 'Single prediction',
            'POST /predict-batch': 'Batch predictions'
        },
        'example_prediction': {
            'electricity_kwh': 450.5,
            'gas_therms': 30.2,
            'distance_km': 500.0,
            'vehicle_type': 'sedan',
            'shopping_carbon_kg': 100.5
        }
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)