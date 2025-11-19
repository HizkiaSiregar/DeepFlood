"""
Main Flask Application for Flood Detection System
ENHANCED: Better Gemini integration, clickable citations, location display
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import json
import time
import threading
import uuid
import tempfile
from datetime import datetime, timezone
import pytz

# Import our custom modules
from config import Config
from model_utils import (
    load_model, 
    generate_gradcam, 
    GeminiPostAnalyzer,
    image_to_base64,
    preprocess_image
)

# Database integration
from database import get_database_instance

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
gemini_analyzer = None
db = None
saved_metrics = {
    'iou': 0.8728,
    'miou': 0.8916,
    'accuracy': 0.9465,
    'precision': 0.9250,
    'recall': 0.9379,
    'f1_score': 0.9314,
    'dice': 0.9244
}

# Progress tracking dictionary
progress_data = {}

def initialize_app():
    """Initialize the application (load model and Gemini)"""
    global model, gemini_analyzer
    
    print("\n" + "="*70)
    print("INITIALIZING FLOOD DETECTION SYSTEM")
    print("="*70)
    
    # Load model
    try:
        model = load_model(app.config['MODEL_PATH'])
        if model is None:
            print("⚠ Warning: Model not loaded!")
        else:
            print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        model = None
    
    # Initialize Gemini - CRITICAL FOR DETAILED REPORTS
    try:
        if not app.config.get('GEMINI_API_KEY'):
            raise ValueError("Gemini API key not found in config")
        
        gemini_analyzer = GeminiPostAnalyzer(app.config['GEMINI_API_KEY'])
        
        if gemini_analyzer and gemini_analyzer.model:
            print("✓ Gemini AI initialized successfully - detailed reports enabled")
        else:
            raise ValueError("Gemini model initialization failed")
            
    except Exception as e:
        print(f"⚠ Gemini initialization FAILED: {e}")
        print("⚠ AI reports will use fallback mode (not detailed)")
        gemini_analyzer = None
    
    print("="*70 + "\n")

def initialize_database():
    """Initialize database connection"""
    global db
    try:
        connection_string = app.config.get('MONGODB_CONNECTION_STRING')
        if connection_string:
            db = get_database_instance(connection_string)
            if hasattr(db, 'client') and db.client:
                print("✅ Database initialized successfully")
                return True
        print("⚠️ Database connection failed - continuing without database")
        return False
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e} - continuing without database")
        db = None
        return False

# Initialize on startup
initialize_app()
initialize_database()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', backbone_name=app.config['CURRENT_BACKBONE'])

@app.route('/about')
def about():
    """Render the about/author page"""
    return render_template('about.html')

def update_progress(task_id, progress, message):
    """Update progress for a specific task"""
    progress_data[task_id] = {
        'progress': progress,
        'message': message
    }

def clean_base64_string(base64_str):
    """Clean and validate base64 string"""
    if not base64_str:
        return ""
    
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    
    base64_str = base64_str.strip()
    
    padding_needed = (4 - len(base64_str) % 4) % 4
    if padding_needed:
        base64_str += '=' * padding_needed
    
    return base64_str

def process_image(task_id, file_path, enable_citations, enable_location):
    """Process image and update progress with optional features"""
    try:
        # STEP 0: Pre-validation with Gemini (5-15%)
        update_progress(task_id, 5, "Validating image with Gemini AI...")
        
        filename = os.path.basename(file_path)
        
        # Pre-validation with Gemini
        if gemini_analyzer:
            update_progress(task_id, 10, "Validating image content...")
            is_valid, confidence, message = gemini_analyzer.validate_flood_image(file_path)
            
            if not is_valid:
                raise ValueError(f"Image validation failed: {message}")
            
            update_progress(task_id, 15, f"✓ {message}")
            print(f"✓ Image validation: {message} (confidence: {confidence:.2f})")
        else:
            update_progress(task_id, 15, "Skipping validation (Gemini not available)")
            print("⚠ Gemini not available - skipping validation")
        
        time.sleep(0.2)
        
        # STEP 1: Preprocessing (15-25%)
        update_progress(task_id, 18, "Preprocessing image...")
        
        img_preprocessed, img_resized = preprocess_image(file_path)
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        
        update_progress(task_id, 25, "Image preprocessed")
        time.sleep(0.2)
        
        # STEP 2: Model Prediction (25-50%)
        update_progress(task_id, 30, "Running DeepLabV3+ model prediction...")
        
        prediction = model.predict(img_preprocessed, verbose=0)
        pred_mask = prediction[0]
        
        update_progress(task_id, 45, "Prediction generated")
        time.sleep(0.2)
        
        # STEP 3: Post-processing (50-65%)
        update_progress(task_id, 50, "Processing prediction mask...")
        
        pred_binary = (pred_mask.squeeze() > 0.5).astype(np.uint8) * 255
        
        total_pixels = pred_binary.shape[0] * pred_binary.shape[1]
        flood_pixels = np.sum(pred_binary == 255)
        flood_percentage = (flood_pixels / total_pixels) * 100
        
        confidence_map = pred_mask.squeeze()
        overall_confidence = np.mean(confidence_map)
        max_confidence = np.max(confidence_map)
        min_confidence = np.min(confidence_map)
        mean_confidence = np.mean(confidence_map)
        
        update_progress(task_id, 60, "Prediction processed")
        time.sleep(0.2)
        
        # STEP 4: Grad-CAM Generation (65-75%)
        update_progress(task_id, 65, "Generating Grad-CAM heatmap...")
        
        try:
            heatmap = generate_gradcam(model, img_preprocessed)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            heatmap = None
        
        update_progress(task_id, 70, "Grad-CAM heatmap generated")
        time.sleep(0.2)
        
        # STEP 5: Create Overlay (75-80%)
        update_progress(task_id, 75, "Creating flood overlay...")
        
        overlay = img_resized.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[pred_binary == 255] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        
        update_progress(task_id, 80, "Overlay created")
        time.sleep(0.2)
        
        # STEP 6: Location Detection (80-85%) - ONLY IF ENABLED
        location_data = None
        location_requested = enable_location  # Track if user requested location
        
        if enable_location:
            update_progress(task_id, 82, "🌍 Detecting location from image...")
            if gemini_analyzer:
                try:
                    print("Starting location detection with Gemini...")
                    location_data = gemini_analyzer.detect_location(file_path)
                    if location_data and location_data.get('found'):
                        print(f"✓ Location detected: {location_data.get('place_name', 'Unknown')}")
                        print(f"  Region: {location_data.get('region', 'Unknown')}")
                        print(f"  Confidence: {location_data.get('confidence', 0):.2f}")
                        update_progress(task_id, 85, f"✓ Location: {location_data.get('place_name', 'Unknown')}")
                    else:
                        print("⚠ Location could not be determined from image")
                        update_progress(task_id, 85, "Location could not be determined")
                        # Set location_data to None so frontend knows not to show section
                        location_data = None
                except Exception as e:
                    print(f"⚠ Location detection error: {e}")
                    location_data = None
                    update_progress(task_id, 85, "Location detection failed")
            else:
                print("⚠ Gemini not available for location detection")
                update_progress(task_id, 85, "Gemini not available for location")
                location_data = None
        else:
            print("Location detection not requested - skipping")
            update_progress(task_id, 85, "Skipping location detection")
            location_data = None
        
        time.sleep(0.2)
        
        # STEP 7: AI Analysis Report (85-95%) - ENHANCED FOR DETAILED REPORTS
        update_progress(task_id, 87, "📝 Generating comprehensive AI analysis with Gemini...")
        
        metrics_for_gemini = {
            'flood_percentage': flood_percentage,
            'flooded_pixels': flood_pixels,
            'total_pixels': total_pixels,
            'prediction_confidence': overall_confidence * 100,
            'max_confidence': max_confidence * 100,
            'min_confidence': min_confidence * 100,
            'mean_confidence': mean_confidence * 100,
            'model_iou': saved_metrics.get('iou', 0),
            'model_miou': saved_metrics.get('miou', 0),
            'model_accuracy': saved_metrics.get('accuracy', 0),
            'model_precision': saved_metrics.get('precision', 0),
            'model_recall': saved_metrics.get('recall', 0),
            'model_f1_score': saved_metrics.get('f1_score', 0),
            'model_dice': saved_metrics.get('dice', 0),
        }
        
        # CRITICAL: Check if Gemini is available
        if gemini_analyzer and gemini_analyzer.model:
            try:
                print(f"\n{'='*60}")
                print(f"GENERATING DETAILED GEMINI REPORT")
                print(f"Citations enabled: {enable_citations}")
                print(f"Flood percentage: {flood_percentage:.2f}%")
                print(f"{'='*60}\n")
                
                gemini_analysis = gemini_analyzer.generate_detailed_report(
                    metrics_for_gemini, 
                    flood_percentage,
                    include_citations=enable_citations
                )
                
                print(f"\n{'='*60}")
                print(f"✓ GEMINI REPORT GENERATED")
                print(f"Length: {len(gemini_analysis)} characters")
                print(f"Contains citations: {'[1]' in gemini_analysis}")
                print(f"{'='*60}\n")
                
                if len(gemini_analysis) < 200:
                    print("⚠ WARNING: Report seems too short! Using fallback...")
                    raise ValueError("Report too short")
                    
            except Exception as e:
                print(f"⚠ Gemini analysis failed: {e}")
                print("Using enhanced fallback report...")
                gemini_analysis = generate_fallback_report(metrics_for_gemini, flood_percentage)
        else:
            print("⚠ Gemini not available - using fallback report")
            gemini_analysis = generate_fallback_report(metrics_for_gemini, flood_percentage)
        
        update_progress(task_id, 95, "✓ AI analysis report completed")
        time.sleep(0.2)
        
        # STEP 8: Finalize (95-100%)
        update_progress(task_id, 98, "Finalizing results...")
        
        original_b64 = clean_base64_string(image_to_base64(img_resized))
        prediction_b64 = clean_base64_string(image_to_base64(pred_binary))
        gradcam_b64 = clean_base64_string(image_to_base64(heatmap)) if heatmap is not None else ''
        overlay_b64 = clean_base64_string(image_to_base64(overlay))
        
        result = {
            'original': original_b64,
            'prediction': prediction_b64,
            'gradcam': gradcam_b64,
            'overlay': overlay_b64,
            
            'flood_percentage': float(flood_percentage),
            'flooded_pixels': int(flood_pixels),
            'total_pixels': int(total_pixels),
            'prediction_confidence': float(overall_confidence * 100),
            'max_confidence': float(max_confidence * 100),
            'min_confidence': float(min_confidence * 100),
            'mean_confidence': float(mean_confidence * 100),
            
            'model_iou': float(saved_metrics.get('iou', 0)),
            'model_miou': float(saved_metrics.get('miou', 0)),
            'model_accuracy': float(saved_metrics.get('accuracy', 0)),
            'model_precision': float(saved_metrics.get('precision', 0)),
            'model_recall': float(saved_metrics.get('recall', 0)),
            'model_f1_score': float(saved_metrics.get('f1_score', 0)),
            'model_dice': float(saved_metrics.get('dice', 0)),
            
            'gemini_analysis': gemini_analysis,
            'location_data': location_data
        }
        
        update_progress(task_id, 100, "✓ Analysis complete!")
        time.sleep(0.2)
        
        # Save to database
        if db and task_id:
            try:
                db.save_analysis_result(task_id, result, filename)
                print(f"✅ Analysis saved to database: {task_id}")
            except Exception as e:
                print(f"⚠️ Could not save to database: {e}")
        
        progress_data[task_id]['result'] = result
        progress_data[task_id]['completed'] = True
        
        try:
            os.remove(file_path)
        except:
            pass
        
        print(f"{'='*70}\n")
        print("✓ Prediction completed successfully")
        
    except Exception as e:
        print(f"\n⚠ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        progress_data[task_id]['error'] = str(e)
        progress_data[task_id]['completed'] = True

def generate_fallback_report(metrics, flood_percentage):
    """Enhanced fallback report when Gemini fails"""
    if flood_percentage < 15:
        severity = "LOW SEVERITY"
        risk = "minimal immediate risk"
        action = "Continue routine monitoring. Implement preventive drainage maintenance."
    elif flood_percentage < 40:
        severity = "MODERATE SEVERITY"
        risk = "moderate risk requiring attention"
        action = "Prepare emergency response protocols. Alert relevant authorities and evacuation teams."
    elif flood_percentage < 70:
        severity = "HIGH SEVERITY"
        risk = "significant risk requiring immediate action"
        action = "Initiate emergency response procedures. Begin evacuation of affected areas."
    else:
        severity = "CRITICAL SEVERITY"
        risk = "extreme danger requiring urgent action"
        action = "IMMEDIATE evacuation required. Deploy emergency services and rescue teams."
    
    return f"""**FLOOD SEVERITY ASSESSMENT - {severity}**

The DeepLabV3+ semantic segmentation model has detected flood coverage of {flood_percentage:.2f}% across the analyzed area, indicating {risk}. The model demonstrates {metrics.get('model_accuracy', 0)*100:.1f}% accuracy with an IoU score of {metrics.get('model_iou', 0):.4f}, suggesting high confidence in these predictions.

**AFFECTED AREA ANALYSIS**

Based on the segmentation results, approximately {flood_percentage:.2f}% of the visible area ({metrics.get('flooded_pixels', 0):,} pixels out of {metrics.get('total_pixels', 0):,} total pixels) shows flood characteristics. The model's prediction confidence averages {metrics.get('prediction_confidence', 0):.2f}%, with peak confidence reaching {metrics.get('max_confidence', 0):.2f}%.

**EMERGENCY RESPONSE RECOMMENDATIONS**

{action} The Grad-CAM visualization highlights the specific regions where the model focused its attention during flood detection, providing explainability for the predictions.

**MODEL RELIABILITY**

This analysis is based on a DeepLabV3+ model achieving {metrics.get('model_accuracy', 0)*100:.1f}% accuracy, {metrics.get('model_precision', 0):.4f} precision, and {metrics.get('model_recall', 0):.4f} recall on validation data, indicating reliable flood detection capabilities."""

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and start processing"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not Config.allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        enable_citations = request.form.get('enable_citations', 'false').lower() == 'true'
        enable_location = request.form.get('enable_location', 'false').lower() == 'true'
        
        print(f"\n{'='*70}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"Optional features:")
        print(f"  - IEEE Citations: {enable_citations}")
        print(f"  - Location Detection: {enable_location}")
        print(f"  - Gemini Available: {gemini_analyzer is not None}")
        print(f"{'='*70}\n")
        
        task_id = str(uuid.uuid4())
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(filepath)
        
        progress_data[task_id] = {
            'progress': 5,
            'message': 'Starting image validation...',
            'completed': False
        }
        
        thread = threading.Thread(
            target=process_image, 
            args=(task_id, filepath, enable_citations, enable_location)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get progress for a specific task"""
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
    
    data = progress_data[task_id].copy()
    return jsonify(data)

# Database API Routes
@app.route('/health')
def health_check():
    """Enhanced health check"""
    status = {
        'status': 'running',
        'model_loaded': model is not None,
        'gemini_available': gemini_analyzer is not None and gemini_analyzer.model is not None,
        'database_available': db is not None and hasattr(db, 'client') and db.client is not None
    }
    return jsonify(status)

@app.route('/api/history')
def get_history():
    """Get analysis history from database"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        limit = request.args.get('limit', 20, type=int)
        analyses = db.get_recent_analyses(limit=limit)
        
        return jsonify({
            'success': True,
            'analyses': analyses,
            'count': len(analyses)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>')
def get_analysis(analysis_id):
    """Get specific analysis by ID with images"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        analysis = db.get_analysis_by_id(analysis_id)
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete analysis by ID"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        success = db.delete_analysis(analysis_id)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-all', methods=['DELETE'])
def delete_all_analyses():
    """Delete all analyses"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        deleted_count = db.delete_all_analyses()
        return jsonify({
            'success': True,
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export')
def export_analyses():
    """Export all analyses to JSON"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            result = db.export_to_json(tmp_file.name)
            
            if result['success']:
                with open(tmp_file.name, 'r') as f:
                    export_data = f.read()
                
                os.unlink(tmp_file.name)
                
                response = Response(
                    export_data,
                    mimetype='application/json',
                    headers={
                        'Content-Disposition': f'attachment; filename=flood_analysis_export_{time.strftime("%Y%m%d")}.json'
                    }
                )
                return response
            else:
                return jsonify({'success': False, 'error': result.get('error', 'Export failed')}), 500
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get database statistics"""
    if not db:
        return jsonify({'success': False, 'error': 'Database not available'}), 503
    
    try:
        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌊 FLOOD DETECTION SYSTEM - LOCAL DEPLOYMENT")
    print("="*70)
    print(f"Server starting on http://localhost:5000")
    print(f"Backbone: {app.config['CURRENT_BACKBONE']}")
    print(f"Model path: {app.config['MODEL_PATH']}")
    print(f"Gemini Status: {'✓ Available' if gemini_analyzer else '✗ Not Available'}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)