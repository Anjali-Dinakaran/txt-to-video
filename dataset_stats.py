dataset_stats.py - Dataset statistics
class DatasetStatistics:
    """Generate dataset statistics"""
    
    def __init__(self, data_dir, caption_file):
        self.data_dir = Path(data_dir)
        self.caption_file = caption_file
        
    def generate_stats(self):
        """Generate comprehensive dataset statistics"""
        stats = {}
        
        # Load captions
        with open(self.caption_file, 'r') as f:
            captions = json.load(f)
        
        stats['total_videos'] = len(captions)
        stats['total_captions'] = len(captions)
        
        # Caption statistics
        caption_lengths = [len(caption.split()) for caption in captions.values()]
        stats['avg_caption_length'] = sum(caption_lengths) / len(caption_lengths)
        stats['min_caption_length'] = min(caption_lengths)
        stats['max_caption_length'] = max(caption_lengths)
        
        # Video statistics
        video_stats = []
        for video_name in captions.keys():
            video_path = self.data_dir / video_name
            if video_path.exists():
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    video_stats.append({
                        'fps': fps,
                        'duration': duration,
                        'frames': frame_count,
                        'resolution': (width, height)
                    })
                except:
                    continue
        
        if video_stats:
            stats['avg_fps'] = sum(v['fps'] for v in video_stats) / len(video_stats)
            stats['avg_duration'] = sum(v['duration'] for v in video_stats) / len(video_stats)
            stats['avg_frames'] = sum(v['frames'] for v in video_stats) / len(video_stats)
            
            resolutions = [v['resolution'] for v in video_stats]
            unique_resolutions = list(set(resolutions))
            stats['resolutions'] = unique_resolutions
        
        return stats
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.generate_stats()
        
        print("="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total videos: {stats['total_videos']}")
        print(f"Total captions: {stats['total_captions']}")
        print(f"Average caption length: {stats['avg_caption_length']:.1f} words")
        print(f"Caption length range: {stats['min_caption_length']}-{stats['max_caption_length']} words")
        
        if 'avg_fps' in stats:
            print(f"\nVideo Statistics:")
            print(f"Average FPS: {stats['avg_fps']:.1f}")
            print(f"Average duration: {stats['avg_duration']:.2f} seconds")
            print(f"Average frames: {stats['avg_frames']:.0f}")
            print(f"Resolutions found: {stats['resolutions']}")
        print("="*50)