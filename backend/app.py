from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from model_engine import LoanMLEngine
import os
import pandas as pd

app = Flask(__name__, 
            static_folder='../static', 
            template_folder='templates')
CORS(app) # Enable CORS for development flexibility

# Initialize the ML Engine
# We write data to backend/data/synthetic_loans.csv and models to backend/models
ml_engine = LoanMLEngine()

# Ensure synthetic dataset is generated and models are pre-trained on server start
if not os.path.exists(ml_engine.data_path):
    print("Initial startup: generating synthetic credit dataset...")
    ml_engine.generate_synthetic_data()

print("Initial startup: training/loading machine learning models...")
# Try loading existing models first to keep startup instant, otherwise train
if not ml_engine.load_cached_models():
    ml_engine.train_models()


@app.route('/')
def index():
    """Serves the main application page."""
    return render_template('index.html')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns evaluation metrics for Logistic Regression, Random Forest, and Gradient Boosting (XGBoost)."""
    try:
        # Refresh or fetch current metrics
        metrics_data = ml_engine.metrics
        
        # Include feature importance lists for visual analytics
        feature_importance = getattr(ml_engine, 'feature_importance_dict', {})
        # Format feature importance for easy charting: list of {feature: name, importance: val}
        sorted_importance = sorted(
            [{'feature': ml_engine._clean_feature_name(k), 'value': round(v, 4)} for k, v in feature_importance.items()],
            key=lambda x: x['value'],
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'metrics': metrics_data,
            'feature_importance': sorted_importance[:10] # Top 10 features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Takes applicant details, processes them, returns risk score (0-100),
    risk tier, approved/declined decision, and game-theoretic SHAP attributions.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
            
        model_name = data.get('model_name', 'Gradient Boosting (XGBoost)')
        
        # Map incoming UI fields to exact model schema fields
        applicant_data = {
            'income': float(data.get('income', 50000)),
            'loan_amount': float(data.get('loan_amount', 15000)),
            'credit_score': int(data.get('credit_score', 650)),
            'credit_utilization': float(data.get('credit_utilization', 30.0)) / 100.0, # convert % to ratio
            'delinquencies': int(data.get('delinquencies', 0)),
            'repayment_score': float(data.get('repayment_score', 75)),
            'employment_years': float(data.get('employment_years', 5)),
            'dti': float(data.get('dti', 25.0)) / 100.0, # convert % to ratio
            'ltv': float(data.get('ltv', 60.0)) / 100.0, # convert % to ratio
            'employment_type': data.get('employment_type', 'Full-time'),
            'loan_purpose': data.get('loan_purpose', 'Debt Consolidation')
        }
        
        # Predict
        result = ml_engine.predict_risk(applicant_data, model_name=model_name)
        return jsonify({
            'success': True,
            'prediction': result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """Serves a slice of the synthetic database for the interactive explorer."""
    try:
        if not os.path.exists(ml_engine.data_path):
            ml_engine.generate_synthetic_data()
            
        df = pd.read_csv(ml_engine.data_path)
        
        # Simple pagination
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 15))
        search = request.args.get('search', '')
        
        # Fill NaNs for JSON serialization
        df_filled = df.fillna({
            'repayment_score': 70.0,
            'employment_years': 3.5,
            'employment_type': 'Full-time'
        })
        
        # Filter if search term is provided
        if search:
            search_lower = search.lower()
            df_filled = df_filled[
                df_filled['employment_type'].astype(str).str.lower().str.contains(search_lower) |
                df_filled['loan_purpose'].astype(str).str.lower().str.contains(search_lower) |
                df_filled['defaulted'].apply(lambda x: 'default' if x == 1 else 'paid').str.contains(search_lower)
            ]
            
        total_rows = len(df_filled)
        
        start = (page - 1) * limit
        end = start + limit
        
        records = df_filled.iloc[start:end].to_dict(orient='records')
        
        # Convert default integer to clean string status for UI
        for r in records:
            r['status'] = 'Defaulted' if r['defaulted'] == 1 else 'Fully Paid'
            r['status_color'] = '#ef4444' if r['defaulted'] == 1 else '#10b981'
            
        return jsonify({
            'success': True,
            'data': records,
            'total': total_rows,
            'page': page,
            'limit': limit,
            'total_pages': (total_rows + limit - 1) // limit
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """Triggers live model re-training and returns refreshed metrics."""
    try:
        # Re-generate synthetic dataset with a minor shift to simulate updates
        ml_engine.generate_synthetic_data(random_seed=42 + int(pd.Timestamp.now().microsecond % 100))
        # Re-train
        metrics_data = ml_engine.train_models()
        
        feature_importance = getattr(ml_engine, 'feature_importance_dict', {})
        sorted_importance = sorted(
            [{'feature': ml_engine._clean_feature_name(k), 'value': round(v, 4)} for k, v in feature_importance.items()],
            key=lambda x: x['value'],
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'message': 'Models successfully re-trained on new data stream.',
            'metrics': metrics_data,
            'feature_importance': sorted_importance[:10]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Fallback routing to serve static files correctly if served on non-standard deployment
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    # Start the server on port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
