# preprocessing.py - Video preprocessing utilities
import subprocess
from pathlib import Path
import tempfile
import shutil

class VideoPreprocessor:
    """Preprocess videos to match required specifications"""
    
    def __init__(self, target_resolution=512, target_fps=24):
        self.target_resolution = target_resolution
        self.target_fps = target_fps
    
    def preprocess_video(self, input_path, output_path):
        """Preprocess a single video"""
        try:
            # Use FFmpeg to resize and adjust FPS
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f'scale={self.target_resolution}:{self.target_resolution}:force_original_aspect_ratio=decrease,pad={self.target_resolution}:{self.target_resolution}:(ow-iw)/2:(oh-ih)/2',
                '-r', str(self.target_fps),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-an',  # Remove audio
                '-y',   # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Preprocessed: {input_path} -> {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error preprocessing {input_path}: {str(e)}")
            return False
    
    def preprocess_dataset(self, input_dir, output_dir, caption_file):
        """Preprocess entire dataset"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load captions
        with open(caption_file, 'r') as f:
            captions = json.load(f)
        
        success_count = 0
        total_count = len(captions)
        
        for video_name in captions.keys():
            input_path = input_dir / video_name
            output_path = output_dir / video_name
            
            if input_path.exists():
                if self.preprocess_video(input_path, output_path):
                    success_count += 1
            else:
                logger.warning(f"Video not found: {input_path}")
        
        logger.info(f"Preprocessing completed: {success_count}/{total_count} videos processed")
        
        # Copy caption file to output directory
        output_caption_file = output_dir.parent / "captions.json"
        shutil.copy2(caption_file, output_caption_file)
        
        return success_count == total_count