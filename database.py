"""
MongoDB Database Integration for Flask Flood Detection App
FIXED: Timezone issues and Base64 encoding + THUMBNAIL IMAGE RETRIEVAL
"""
import os
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
import gridfs
import base64
import json
from typing import Optional, Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your local timezone here - CHANGE THIS TO YOUR TIMEZONE
LOCAL_TIMEZONE = timezone(timedelta(hours=8))  # Example: Philippines (UTC+8)
# For other timezones:
# UTC+0: timezone.utc
# UTC+1: timezone(timedelta(hours=1))
# UTC-5: timezone(timedelta(hours=-5))
# etc.

class FloodDetectionDB:
    """MongoDB database handler for flood detection system"""
    
    def __init__(self, connection_string: str = None, database_name: str = "flood_detection_db"):
        """Initialize MongoDB connection"""
        self.connection_string = connection_string or "mongodb://localhost:27017/"
        self.database_name = database_name
        self.client = None
        self.db = None
        self.fs = None
        self.connect()
    
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.fs = gridfs.GridFS(self.db)
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Analysis results collection indexes
            self.db.analysis_results.create_index("task_id", unique=True)
            self.db.analysis_results.create_index("timestamp")
            self.db.analysis_results.create_index("flood_percentage")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")
    
    def _get_local_time(self):
        """Get current time in local timezone"""
        return datetime.now(LOCAL_TIMEZONE)

    def save_analysis_result(self, task_id: str, result_data: Dict, filename: str = None) -> str:
        """Save flood analysis result to database"""
        try:
            # Get local timestamp
            local_time = self._get_local_time()
            
            # Prepare document for database
            document = {
                "task_id": task_id,
                "timestamp": local_time,  # Store in local time
                "filename": filename or f"Analysis_{local_time.strftime('%Y%m%d_%H%M%S')}",
                
                # Analysis metrics
                "flood_percentage": result_data.get("flood_percentage", 0),
                "flooded_pixels": result_data.get("flooded_pixels", 0),
                "total_pixels": result_data.get("total_pixels", 0),
                "prediction_confidence": result_data.get("prediction_confidence", 0),
                "max_confidence": result_data.get("max_confidence", 0),
                "min_confidence": result_data.get("min_confidence", 0),
                "mean_confidence": result_data.get("mean_confidence", 0),
                
                # Model performance metrics
                "model_metrics": {
                    "iou": result_data.get("model_iou", 0),
                    "miou": result_data.get("model_miou", 0),
                    "accuracy": result_data.get("model_accuracy", 0),
                    "precision": result_data.get("model_precision", 0),
                    "recall": result_data.get("model_recall", 0),
                    "f1_score": result_data.get("model_f1_score", 0),
                    "dice": result_data.get("model_dice", 0)
                },
                
                # AI analysis
                "gemini_analysis": result_data.get("gemini_analysis", ""),
                "location_data": result_data.get("location_data"),
                
                # Severity classification
                "severity_level": self._classify_flood_severity(result_data.get("flood_percentage", 0)),
                
                # Processing metadata
                "processing_completed": True,
                "created_at": local_time
            }
            
            # Save images to GridFS and get their IDs - FIXED
            image_ids = self._save_images_to_gridfs(task_id, result_data, filename)
            document["image_ids"] = image_ids
            
            # Insert document
            result = self.db.analysis_results.insert_one(document)
            doc_id = str(result.inserted_id)
            
            logger.info(f"✅ Analysis result saved: {task_id} -> {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Error saving analysis result: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_images_to_gridfs(self, task_id: str, result_data: Dict, filename: str = None) -> Dict[str, str]:
        """Save base64 encoded images to GridFS - FIXED for proper base64 handling"""
        try:
            saved_images = {}
            
            image_types = ['original', 'prediction', 'gradcam', 'overlay']
            
            for image_type in image_types:
                base64_data = result_data.get(image_type)
                if base64_data:
                    try:
                        # Clean base64 string - remove data URL prefix if present
                        if isinstance(base64_data, str):
                            # Remove data URL prefix
                            if ',' in base64_data:
                                base64_data = base64_data.split(',', 1)[1]
                            
                            # Remove whitespace
                            base64_data = base64_data.strip()
                            
                            # Add padding if needed
                            padding_needed = (4 - len(base64_data) % 4) % 4
                            if padding_needed:
                                base64_data += '=' * padding_needed
                            
                            # Decode base64 image
                            image_data = base64.b64decode(base64_data)
                            
                            # Create filename for GridFS
                            gridfs_filename = f"{task_id}_{image_type}.png"
                            
                            # Save to GridFS
                            file_id = self.fs.put(
                                image_data,
                                filename=gridfs_filename,
                                content_type="image/png",
                                metadata={
                                    "task_id": task_id,
                                    "image_type": image_type,
                                    "original_filename": filename,
                                    "upload_timestamp": self._get_local_time()
                                }
                            )
                            
                            saved_images[image_type] = str(file_id)
                            logger.info(f"✅ Saved {image_type} image to GridFS: {file_id}")
                        
                    except Exception as img_error:
                        logger.warning(f"⚠️ Could not save {image_type} image: {img_error}")
            
            return saved_images
            
        except Exception as e:
            logger.error(f"❌ Error saving images: {e}")
            return {}
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict]:
        """Get recent flood analyses for history display WITH THUMBNAIL IMAGES"""
        try:
            cursor = self.db.analysis_results.find().sort("timestamp", -1).limit(limit)
            results = []
            
            for doc in cursor:
                # Format timestamp for display - using stored local time
                timestamp = doc.get("timestamp")
                if timestamp:
                    # If timestamp is timezone-aware, convert to local
                    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo:
                        timestamp = timestamp.astimezone(LOCAL_TIMEZONE)
                    date_str = timestamp.strftime('%m/%d/%Y')
                    time_str = timestamp.strftime('%I:%M %p')
                else:
                    date_str = "Unknown Date"
                    time_str = "Unknown Time"
                
                # Get images as base64 for thumbnail display - CRITICAL FIX
                image_ids = doc.get("image_ids", {})
                images = {}
                
                # Only retrieve original and prediction for thumbnails (saves bandwidth)
                for img_type in ['original', 'prediction']:
                    if img_type in image_ids:
                        file_id = image_ids[img_type]
                        img_data = self.get_image_as_base64(file_id)
                        if img_data:
                            images[img_type] = img_data
                        else:
                            logger.warning(f"⚠️ Could not retrieve {img_type} image for {doc['_id']}")
                
                # Convert ObjectId to string and format for frontend
                analysis = {
                    "_id": str(doc["_id"]),
                    "task_id": doc.get("task_id", ""),
                    "timestamp": doc.get("timestamp"),
                    "date_display": date_str,
                    "time_display": time_str,
                    "filename": doc.get("filename", f"Analysis {date_str}"),
                    "flood_percentage": round(doc.get("flood_percentage", 0), 1),
                    "severity_level": doc.get("severity_level", "unknown"),
                    "gemini_analysis": doc.get("gemini_analysis", "")[:80] + "..." if len(doc.get("gemini_analysis", "")) > 80 else doc.get("gemini_analysis", ""),
                    "location": doc.get("location_data", {}).get("place_name", "Unknown") if doc.get("location_data") else "Unknown",
                    "image_ids": image_ids,
                    "images": images  # CRITICAL: Base64 images for thumbnails
                }
                results.append(analysis)
            
            logger.info(f"✅ Retrieved {len(results)} analyses with thumbnails")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving recent analyses: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """Get analysis result by document ID with images"""
        try:
            doc = self.db.analysis_results.find_one({"_id": ObjectId(analysis_id)})
            if doc:
                # Convert for frontend display
                doc["_id"] = str(doc["_id"])
                
                # Get images as base64 for display
                image_ids = doc.get("image_ids", {})
                images = {}
                for img_type, file_id in image_ids.items():
                    img_data = self.get_image_as_base64(file_id)
                    if img_data:
                        images[img_type] = img_data
                
                doc["images"] = images
                return doc
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving analysis: {e}")
            return None
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis and associated images"""
        try:
            # Get the analysis first to find associated images
            analysis = self.db.analysis_results.find_one({"_id": ObjectId(analysis_id)})
            if not analysis:
                return False
            
            # Delete associated images from GridFS
            image_ids = analysis.get("image_ids", {})
            for image_type, file_id in image_ids.items():
                try:
                    self.fs.delete(ObjectId(file_id))
                except Exception as img_error:
                    logger.warning(f"⚠️ Could not delete image {image_type}: {img_error}")
            
            # Delete the analysis document
            result = self.db.analysis_results.delete_one({"_id": ObjectId(analysis_id)})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Analysis deleted: {analysis_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting analysis: {e}")
            return False
    
    def delete_all_analyses(self) -> int:
        """Delete all analyses and images (clear history)"""
        try:
            # Delete all GridFS files
            for grid_file in self.fs.find():
                try:
                    self.fs.delete(grid_file._id)
                except Exception:
                    pass
            
            # Delete all analysis documents
            result = self.db.analysis_results.delete_many({})
            deleted_count = result.deleted_count
            
            logger.info(f"✅ Deleted all {deleted_count} analyses")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error deleting all analyses: {e}")
            return 0
    
    def get_image_as_base64(self, file_id: str) -> Optional[str]:
        """Retrieve image from GridFS and return as base64"""
        try:
            file_id_obj = ObjectId(file_id)
            gridfs_file = self.fs.get(file_id_obj)
            image_data = gridfs_file.read()
            
            # Convert to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return base64_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving image: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            total_analyses = self.db.analysis_results.count_documents({})
            
            if total_analyses == 0:
                return {
                    "total_analyses": 0,
                    "avg_flood_percentage": 0,
                    "severity_distribution": {"low": 0, "moderate": 0, "high": 0, "severe": 0}
                }
            
            # Calculate average flood percentage
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_flood": {"$avg": "$flood_percentage"}
                }}
            ]
            avg_result = list(self.db.analysis_results.aggregate(pipeline))
            avg_flood = round(avg_result[0]["avg_flood"], 2) if avg_result else 0
            
            # Get severity distribution
            severity_distribution = {"low": 0, "moderate": 0, "high": 0, "severe": 0}
            severity_pipeline = [
                {"$group": {
                    "_id": "$severity_level",
                    "count": {"$sum": 1}
                }}
            ]
            severity_results = list(self.db.analysis_results.aggregate(severity_pipeline))
            for item in severity_results:
                if item["_id"] in severity_distribution:
                    severity_distribution[item["_id"]] = item["count"]
            
            return {
                "total_analyses": total_analyses,
                "avg_flood_percentage": avg_flood,
                "severity_distribution": severity_distribution
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {"total_analyses": 0, "avg_flood_percentage": 0, "severity_distribution": {"low": 0, "moderate": 0, "high": 0, "severe": 0}}
    
    def export_to_json(self, output_file: str = None) -> Dict:
        """Export all analysis data to JSON"""
        try:
            if not output_file:
                timestamp = self._get_local_time().strftime('%Y%m%d_%H%M%S')
                output_file = f"flood_analysis_export_{timestamp}.json"
            
            export_data = {
                "export_timestamp": self._get_local_time().isoformat(),
                "total_analyses": self.db.analysis_results.count_documents({}),
                "analyses": []
            }
            
            # Export analysis results
            for doc in self.db.analysis_results.find().sort("timestamp", -1):
                doc["_id"] = str(doc["_id"])
                if doc.get("timestamp"):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                if doc.get("created_at"):
                    doc["created_at"] = doc["created_at"].isoformat()
                export_data["analyses"].append(doc)
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"✅ Data exported to: {output_file}")
            return {"success": True, "filename": output_file, "count": len(export_data["analyses"])}
            
        except Exception as e:
            logger.error(f"❌ Error exporting data: {e}")
            return {"success": False, "error": str(e)}
    
    def _classify_flood_severity(self, flood_percentage: float) -> str:
        """Classify flood severity based on percentage"""
        if flood_percentage < 10:
            return "low"
        elif flood_percentage < 30:
            return "moderate"
        elif flood_percentage < 60:
            return "high"
        else:
            return "severe"
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Singleton instance
db_instance = None

def get_database_instance(connection_string: str = None) -> FloodDetectionDB:
    """Get or create database instance"""
    global db_instance
    if db_instance is None:
        db_instance = FloodDetectionDB(connection_string)
    return db_instance

def test_connection(connection_string: str) -> bool:
    """Test MongoDB connection"""
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        client.close()
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False