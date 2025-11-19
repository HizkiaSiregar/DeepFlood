"""
Model utilities for loading and making predictions
ENHANCED: REAL citations from Gemini, proper location gating
"""
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, Xception
import google.generativeai as genai
from PIL import Image
import base64
from io import BytesIO

# Register custom objects
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CUSTOM LAYERS AND METRICS (keeping your original training code)
# ============================================================================

class CastToFloat32(layers.Layer):
    """Custom layer to cast tensors to float32"""
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
    
    def call(self, x):
        return tf.cast(x, tf.float32)
    
    def get_config(self):
        config = super().get_config()
        config.pop('dtype', None)
        return config

class EnhancedConvBlock(layers.Layer):
    """Enhanced convolution block"""
    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.1, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.conv = None
        self.bn = None
        self.activation = None
        self.dropout = None
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.filters, self.kernel_size, padding='same',
            dilation_rate=self.dilation_rate, dtype='float32'
        )
        self.bn = layers.BatchNormalization(dtype='float32')
        self.activation = layers.Activation('relu', dtype='float32')
        self.dropout = layers.Dropout(self.dropout_rate, dtype='float32')
        super().build(input_shape)
    
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.pop('dtype', None)
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config

class DynamicUpsampling(layers.Layer):
    """Custom layer for dynamic upsampling"""
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
    
    def call(self, inputs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(source, [target_shape[1], target_shape[2]], method='bilinear')
    
    def get_config(self):
        config = super().get_config()
        config.pop('dtype', None)
        return config

class GlobalPoolingBranch(layers.Layer):
    """Custom layer for global pooling branch"""
    def __init__(self, filters=256, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
        self.filters = filters
        self.global_pool = None
        self.dense = None
        self.dropout = None
        self.reshape = None
    
    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D(dtype='float32')
        self.dense = layers.Dense(self.filters, activation='relu', dtype='float32')
        self.dropout = layers.Dropout(0.1, dtype='float32')
        self.reshape = layers.Reshape((1, 1, self.filters))
        super().build(input_shape)
    
    def call(self, inputs):
        x, target_tensor = inputs
        pooled = self.global_pool(x)
        pooled = self.dense(pooled)
        pooled = self.dropout(pooled)
        pooled = self.reshape(pooled)
        target_shape = tf.shape(target_tensor)
        upsampled = tf.image.resize(pooled, [target_shape[1], target_shape[2]], method='bilinear')
        return upsampled
    
    def get_config(self):
        config = super().get_config()
        config.pop('dtype', None)
        config.update({'filters': self.filters})
        return config

class FinalResize(layers.Layer):
    """Custom layer for final resize"""
    def __init__(self, target_height, target_width, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
    
    def call(self, x):
        return tf.image.resize(x, (self.target_height, self.target_width))
    
    def get_config(self):
        config = super().get_config()
        config.pop('dtype', None)
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

# ============================================================================
# LOSS FUNCTIONS AND METRICS
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    """Focal loss to handle class imbalance"""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1)) \
           -tf.keras.backend.mean((1-alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0))

def combined_loss(y_true, y_pred):
    """Combined loss: BCE + Dice + Focal"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return 0.4 * bce + 0.4 * dice + 0.2 * focal

def iou_metric(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def miou_metric(y_true, y_pred, smooth=1e-6):
    """mIoU (mean IoU)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection_1 = tf.reduce_sum(y_true * y_pred)
    union_1 = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection_1
    iou_1 = (intersection_1 + smooth) / (union_1 + smooth)
    y_true_0 = 1 - y_true
    y_pred_0 = 1 - y_pred
    intersection_0 = tf.reduce_sum(y_true_0 * y_pred_0)
    union_0 = tf.reduce_sum(y_true_0) + tf.reduce_sum(y_pred_0) - intersection_0
    iou_0 = (intersection_0 + smooth) / (union_0 + smooth)
    return (iou_1 + iou_0) / 2.0

def precision_metric(y_true, y_pred, smooth=1e-6):
    """Precision metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall_metric(y_true, y_pred, smooth=1e-6):
    """Recall metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    actual_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (actual_positives + smooth)

def f1_score_metric(y_true, y_pred):
    """F1 Score metric"""
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path):
    """Load trained model with custom objects"""
    custom_objects = {
        'CastToFloat32': CastToFloat32,
        'EnhancedConvBlock': EnhancedConvBlock,
        'DynamicUpsampling': DynamicUpsampling,
        'GlobalPoolingBranch': GlobalPoolingBranch,
        'FinalResize': FinalResize,
        'combined_loss': combined_loss,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'focal_loss': focal_loss,
        'iou_metric': iou_metric,
        'miou_metric': miou_metric,
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_score_metric': f1_score_metric
    }
    
    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss=combined_loss,
                metrics=[iou_metric, miou_metric, precision_metric, 
                        recall_metric, f1_score_metric, dice_coefficient]
            )
            print("✓ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            return None
    
    print("⚠ Model file not found")
    return None

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

def generate_gradcam(model, image):
    """Generate Grad-CAM heatmap"""
    try:
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, layers.Conv2D):
                        last_conv_layer_name = sublayer.name
                        break
                if last_conv_layer_name:
                    break
        
        if last_conv_layer_name is None:
            return None
        
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_batch)
            loss = tf.reduce_mean(predictions)
        
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        heatmap = heatmap.numpy()
        target_size = (image.shape[1] if len(image.shape) == 4 else image.shape[0],
                      image.shape[2] if len(image.shape) == 4 else image.shape[1])
        heatmap = cv2.resize(heatmap, target_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    except Exception as e:
        print(f"⚠ Grad-CAM error: {e}")
        return None

# ============================================================================
# GEMINI AI ANALYZER - WITH REAL CITATIONS
# ============================================================================

class GeminiPostAnalyzer:
    """Gemini AI for detailed analysis with REAL citations from Gemini search"""
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ Gemini AI initialized with model: gemini-2.0-flash-exp")
        except Exception as e:
            print(f"⚠ Gemini initialization failed: {e}")
            self.model = None
    
    def validate_flood_image(self, image_path):
        """Pre-validate if image contains flood or water"""
        if self.model is None:
            return (True, 1.0, "Validation skipped - proceeding with analysis")
        
        try:
            img = Image.open(image_path)
            
            prompt = """Analyze this image and determine if it contains ANY evidence of flooding or water bodies.

Look for: standing water, flooded streets/buildings, rivers, lakes, water bodies, waterlogged areas, puddles, water accumulation, or flood indicators.

Respond in this EXACT format:
CONTAINS_FLOOD: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation in one sentence]

Be generous - if there's ANY water visible or potential flood indicators, say YES."""
            
            response = self.model.generate_content([prompt, img])
            response_text = response.text.strip()
            
            lines = response_text.split('\n')
            contains_flood = False
            confidence = 0.5
            reason = "Unable to determine"
            
            for line in lines:
                if 'CONTAINS_FLOOD:' in line:
                    contains_flood = 'YES' in line.upper()
                elif 'CONFIDENCE:' in line:
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        confidence = 0.7 if contains_flood else 0.3
                elif 'REASON:' in line:
                    reason = line.split(':', 1)[1].strip()
            
            if contains_flood:
                return (True, confidence, f"Image validated: {reason}")
            else:
                return (False, confidence, f"No flood detected: {reason}")
                
        except Exception as e:
            print(f"⚠ Image validation error: {e}")
            return (True, 0.5, "Validation error - proceeding with analysis")
    
    def detect_location(self, image_path):
        """Detect location/landmarks in the image"""
        if self.model is None:
            return None
        
        try:
            img = Image.open(image_path)
            
            prompt = """Analyze this flood image and identify the LOCATION or AREA where this flood occurred.

Look for:
- Recognizable landmarks (buildings, bridges, monuments, signs, infrastructure)
- Geographic features (mountains, coastlines, rivers, terrain)
- Street signs, business names, text, banners, or visible writing
- Architectural styles indicating region/country
- Vegetation, climate indicators, or environmental clues
- ANY visual clues about the location

Respond in this EXACT format:
LOCATION_FOUND: [YES/NO]
CONFIDENCE: [0.0-1.0]
PLACE_NAME: [Specific location name if found, or "Unknown"]
REGION: [City, State/Province, Country if identifiable, or "Unknown"]
COORDINATES_ESTIMATE: [Approximate lat,long if confident, or "Unknown"]
LANDMARKS: [Comma-separated list of identifiable landmarks, or "None"]
REASONING: [Detailed explanation of how you determined the location]

Be thorough but honest - if you cannot identify the location with confidence, say LOCATION_FOUND: NO"""
            
            response = self.model.generate_content([prompt, img])
            response_text = response.text.strip()
            
            location_data = {
                'found': False,
                'confidence': 0.0,
                'place_name': 'Unknown',
                'region': 'Unknown',
                'coordinates': None,
                'landmarks': [],
                'reasoning': ''
            }
            
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if 'LOCATION_FOUND:' in line:
                    location_data['found'] = 'YES' in line.upper()
                elif 'CONFIDENCE:' in line:
                    try:
                        location_data['confidence'] = float(line.split(':', 1)[1].strip())
                    except:
                        pass
                elif 'PLACE_NAME:' in line:
                    location_data['place_name'] = line.split(':', 1)[1].strip()
                elif 'REGION:' in line:
                    location_data['region'] = line.split(':', 1)[1].strip()
                elif 'COORDINATES_ESTIMATE:' in line:
                    coord_str = line.split(':', 1)[1].strip()
                    if coord_str.lower() != 'unknown' and ',' in coord_str:
                        try:
                            lat, lon = coord_str.split(',')
                            location_data['coordinates'] = {
                                'lat': float(lat.strip()),
                                'lng': float(lon.strip())
                            }
                        except:
                            pass
                elif 'LANDMARKS:' in line:
                    landmarks_str = line.split(':', 1)[1].strip()
                    if landmarks_str and landmarks_str.lower() not in ['none', 'unknown']:
                        location_data['landmarks'] = [l.strip() for l in landmarks_str.split(',')]
                elif 'REASONING:' in line:
                    location_data['reasoning'] = line.split(':', 1)[1].strip()
            
            return location_data if location_data['found'] else None
            
        except Exception as e:
            print(f"⚠ Location detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_detailed_report(self, metrics, flood_percentage, include_citations=False):
        """Generate comprehensive flood analysis report with REAL CITATIONS from Gemini"""
        if self.model is None:
            print("⚠ Gemini model not available - using fallback")
            return self._generate_fallback_report(metrics, flood_percentage)
        
        try:
            # Build enhanced prompt for detailed analysis
            citation_section = ""
            if include_citations:
                citation_section = """
CRITICAL CITATION REQUIREMENTS - YOU MUST USE REAL, ACTUAL SOURCES:

1. YOU MUST SEARCH FOR AND CITE REAL ACADEMIC PAPERS AND SOURCES
2. DO NOT MAKE UP OR FABRICATE ANY REFERENCES
3. Include 5-7 citations numbered [1], [2], [3], [4], [5], [6], [7] throughout your analysis
4. Place citations IMMEDIATELY after relevant statements

TOPICS TO SEARCH AND CITE (find REAL sources for these):
- Deep learning for flood detection and segmentation (IEEE, arXiv papers)
- Flood severity classification standards (NOAA, FEMA, WHO)
- Health risks from flooding (CDC, WHO publications)
- Emergency response protocols (FEMA, disaster management journals)
- Remote sensing and satellite imagery for floods (IEEE Geoscience journals)
- Impact of climate change on flooding (Nature, Science journals)

MANDATORY FORMAT - After your analysis, add a "References:" section:

References:

[1] First Author, Second Author, "Paper Title," Journal/Conference Name, vol. X, no. Y, pp. Z-Z, Year. https://doi.org/...
[2] Organization Name, "Document Title," Website, Publication Date. https://www.website.com/path
[3] ... continue for all citations

IMPORTANT RULES:
- ONLY cite sources that you have ACTUALLY FOUND through search or that you KNOW exist
- Include REAL, WORKING URLs or DOIs
- Use a mix of:
  * IEEE journal papers (IEEE Trans. Geoscience, IEEE Access, etc.)
  * Government sources (FEMA.gov, NOAA.gov, CDC.gov, WHO.int)
  * Academic papers (Nature, Science, Remote Sensing journals)
- DO NOT fabricate authors, titles, or DOIs
- If you cannot find a specific source, skip that citation rather than making it up

SEARCH for real papers on: "deep learning flood detection", "semantic segmentation disaster", "flood risk assessment", "emergency flood response"
"""
            
            prompt = f"""You are an expert flood analysis AI system. Generate a COMPREHENSIVE, DETAILED professional flood assessment report for THIS SPECIFIC IMAGE.

IMAGE ANALYSIS RESULTS FOR THIS SPECIFIC IMAGE:
- Flood Coverage: {flood_percentage:.2f}% of analyzed area
- Flooded Pixels: {metrics.get('flooded_pixels', 0):,} out of {metrics.get('total_pixels', 0):,} total pixels
- Prediction Confidence: {metrics.get('prediction_confidence', 0):.2f}%
- Maximum Confidence: {metrics.get('max_confidence', 0):.2f}%
- Mean Confidence: {metrics.get('mean_confidence', 0):.2f}%

MODEL PERFORMANCE METRICS (from validation data):
- IoU: {metrics.get('model_iou', 0):.4f}
- mIoU: {metrics.get('model_miou', 0):.4f}  
- Accuracy: {metrics.get('model_accuracy', 0)*100:.2f}%
- Precision: {metrics.get('model_precision', 0):.4f}
- Recall: {metrics.get('model_recall', 0):.4f}
- F1-Score: {metrics.get('model_f1_score', 0):.4f}

{citation_section}

REQUIRED REPORT STRUCTURE (write 4-5 DETAILED paragraphs):

**1. FLOOD SEVERITY ASSESSMENT**
Classify severity based on {flood_percentage:.2f}% coverage for THIS image. Explain what this percentage means in practical terms. Describe the extent and distribution of flooding visible.

**2. CONFIDENCE AND RELIABILITY ANALYSIS**
Evaluate the model's confidence in THIS specific prediction ({metrics.get('prediction_confidence', 0):.2f}%). Discuss the variation in confidence scores (max: {metrics.get('max_confidence', 0):.2f}%, mean: {metrics.get('mean_confidence', 0):.2f}%). Reference the model's validation performance ({metrics.get('model_accuracy', 0)*100:.1f}% accuracy).

**3. AFFECTED AREA CHARACTERISTICS**
Describe the extent and pattern of flooding detected in THIS image. Analyze which areas show highest confidence for flood detection. Discuss implications for the affected region.

**4. EMERGENCY RESPONSE RECOMMENDATIONS**
Provide SPECIFIC actions based on THIS image's {flood_percentage:.2f}% flood severity. Include immediate actions, medium-term responses, and recovery guidance. Tailor recommendations to the severity level.

**5. MODEL RELIABILITY AND LIMITATIONS**
Briefly mention the DeepLabV3+ model achieved {metrics.get('model_accuracy', 0)*100:.1f}% accuracy. Note that Grad-CAM provides explainability for predictions. Mention any limitations or considerations.

Write in professional, technical language suitable for emergency management officials and disaster response teams. Each paragraph should be 3-5 sentences. Focus on THIS SPECIFIC IMAGE's results.

CRITICAL: 
- Write DETAILED analysis (800+ words total), not brief summaries
- Use specific numbers and metrics from THIS image
- Be professional but accessible
- If citations are enabled, YOU MUST SEARCH FOR AND USE ONLY REAL SOURCES
- DO NOT MAKE UP FAKE REFERENCES - only cite sources you have actually found or know exist with real URLs"""
            
            generation_config = {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
            
            print(f"Sending detailed prompt to Gemini...")
            print(f"Citations enabled: {include_citations}")
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            result = response.text.strip()
            
            print(f"✓ Received Gemini response (length: {len(result)} chars)")
            
            # Validate response length
            if len(result) < 300:
                print("⚠ Response too short, trying again with temperature adjustment...")
                generation_config['temperature'] = 0.7
                response = self.model.generate_content(prompt, generation_config=generation_config)
                result = response.text.strip()
            
            # If citations were requested, verify they exist
            if include_citations:
                if '[1]' not in result or 'References:' not in result:
                    print("⚠ Citations requested but not found in response")
                else:
                    print("✓ Citations found in response")
            
            return result
            
        except Exception as e:
            print(f"⚠ Gemini analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_report(metrics, flood_percentage)
    
    def _generate_fallback_report(self, metrics, flood_percentage):
        """Fallback report if Gemini fails"""
        if flood_percentage < 15:
            severity = "LOW SEVERITY"
            risk = "minimal risk"
            action = "Continue routine monitoring"
        elif flood_percentage < 40:
            severity = "MODERATE SEVERITY"
            risk = "moderate risk requiring attention"
            action = "Prepare emergency response protocols"
        elif flood_percentage < 70:
            severity = "HIGH SEVERITY"
            risk = "significant risk requiring immediate action"
            action = "Initiate emergency response procedures"
        else:
            severity = "CRITICAL SEVERITY"
            risk = "extreme danger requiring urgent action"
            action = "Immediate evacuation and emergency response required"
        
        return f"""**FLOOD SEVERITY ASSESSMENT - {severity}**

The DeepLabV3+ model has detected flood coverage of {flood_percentage:.2f}% across the analyzed area, indicating {risk}. The model demonstrates {metrics.get('model_accuracy', 0)*100:.1f}% accuracy with an IoU score of {metrics.get('model_iou', 0):.4f}, suggesting high confidence in these predictions.

**AFFECTED AREA ANALYSIS**

Based on the segmentation results, approximately {flood_percentage:.2f}% of the visible area shows flood characteristics. The prediction confidence averages {metrics.get('prediction_confidence', 0):.2f}%, indicating reliable detection.

**EMERGENCY RESPONSE RECOMMENDATIONS**

{action}. The Grad-CAM visualization highlights the specific regions where the model focused its attention during flood detection.

**MODEL RELIABILITY**

The analysis is based on a model achieving {metrics.get('model_accuracy', 0)*100:.1f}% accuracy, {metrics.get('model_precision', 0):.4f} precision, and {metrics.get('model_recall', 0):.4f} recall on validation data."""

# ============================================================================
# IMAGE UTILITY FUNCTIONS
# ============================================================================

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    try:
        if img_array.max() <= 1:
            img_array = (img_array * 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array.astype(np.uint8))
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def preprocess_image(image_path, target_size=256):
    """Load and preprocess image for prediction"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized, img_resized