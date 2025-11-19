"""
Configuration file for Flood Detection Flask Application
"""
import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model settings
    MODEL_PATH = 'models/Xception/final_model.keras'  # Path to your trained model
    CURRENT_BACKBONE = 'Xception'
    IMAGE_SIZE = 256
    
    # Gemini AI settings - UPDATED WITH NEW KEY
    GEMINI_API_KEY = "AIzaSyAXS5eNdQr40aE0ev2tDUMdT_9zE-8Sjjs"
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # ============================================================================
    # DATABASE SETTINGS
    # ============================================================================
    
    # MongoDB Connection
    MONGODB_CONNECTION_STRING = os.environ.get('MONGODB_CONNECTION_STRING') or 'mongodb+srv://floodapp:4nvOttXc6pW8fKwc@flood-detection-cluster.ojye3mk.mongodb.net/?appName=flood-detection-cluster'
    
    # Database settings
    DATABASE_NAME = 'flood_detection_db'
    ENABLE_DATABASE = True  # Set to False to disable database features
    
    # ============================================================================
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS