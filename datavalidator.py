# data_validator.py - Validate dataset
import os
import json
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validate video dataset before training"""
    
    def __init__(self, data_dir, caption_file):
        self.data_dir = Path(data_dir)
        self.caption_file = caption_file
        
    def validate(self):
        """Run all validation checks"""
        logger.info("Starting dataset validation...")
        
        issues = []
        issues.extend(self.check_directory_structure())
        issues.extend(self.check_caption_file())
        issues.extend(self.check_videos())
        
        if issues:
            logger.error(f"Found {len(issues)} issues:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("Dataset validation passed!")
            return True
    
    def check_directory_structure(self):
        """Check if directories exist"""
        issues = []
        
        if not self.data_dir.exists():
            issues.append(f"Data directory {self.data_dir} does not exist")
        
        if not Path(self.caption_file).exists():
            issues.append(f"Caption file {self.caption_file} does not exist")
        
        return issues
    
    def check_caption_file(self):
        """Check caption file format"""
        issues = []
        
        try:
            with open(self.caption_file, 'r') as f:
                captions = json.load(f)
            
            if not isinstance(captions, dict):
                issues.append("Caption file should contain a JSON object")
                return issues
            
            if len(captions) == 0:
                issues.append("Caption file is empty")
            
            # Check each caption
            for video_name, caption in captions.items():
                if not isinstance(caption, str):
                    issues.append(f"Caption for {video_name} is not a string")
                
                if len(caption.strip()) == 0:
                    issues.append(f"Caption for {video_name} is empty")
        
        except json.JSONDecodeError:
            issues.append("Caption file is not valid JSON")
        except Exception as e:
            issues.append(f"Error reading caption file: {str(e)}")
        
        return issues
    
    def check_videos(self):
        """Check video files"""
        issues = []
        
        try:
            with open(self.caption_file, 'r') as f:
                captions = json.load(f)
        except:
            return ["Cannot read caption file for video validation"]
        
        for video_name in captions.keys():
            video_path = self.data_dir / video_name
            
            if not video_path.exists():
                issues.append(f"Video file {video_name} does not exist")
                continue
            
            # Check video properties
            try:
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    issues.append(f"Cannot open video file {video_name}")
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.release()
                
                # Check properties
                if frame_count == 0:
                    issues.append(f"Video {video_name} has no frames")
                
                if fps <= 0:
                    issues.append(f"Video {video_name} has invalid FPS: {fps}")
                
                if width != 512 or height != 512:
                    logger.warning(f"Video {video_name} resolution {width}x{height} != 512x512 (will be resized)")
                
                duration = frame_count / fps if fps > 0 else 0
                if duration < 0.5:  # Less than 0.5 seconds
                    issues.append(f"Video {video_name} is too short: {duration:.2f}s")
                    
            except Exception as e:
                issues.append(f"Error checking video {video_name}: {str(e)}")
        
        return issues
