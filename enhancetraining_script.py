#!/usr/bin/env python3
"""
Enhanced Text-to-Video Fine-tuning Script
Complete implementation with error handling, monitoring, and utilities
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import subprocess
import time
import psutil
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Import required libraries
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import StableVideoDiffusionPipeline, UNet3DConditionModel
    from diffusers.schedulers import DDPMScheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as TF
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EnhancedVideoDataset(Dataset):
    """Enhanced dataset with better error handling and preprocessing"""
    
    def __init__(self, data_dir, caption_file, video_length=16, resolution=512, fps=24, validate=True):
        self.data_dir = Path(data_dir)
        self.video_length = video_length
        self.resolution = resolution
        self.fps = fps
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load and validate captions
        self.captions = self._load_captions(caption_file)
        
        # Find valid video files
        self.video_files = self._find_valid_videos()
        
        if len(self.video_files) == 0:
            raise ValueError("No valid video files found!")
        
        self.logger.info(f"Dataset initialized with {len(self.video_files)} videos")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Validate dataset if requested
        if validate:
            self._validate_dataset()
    
    def _load_captions(self, caption_file):
        """Load and validate caption file"""
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            
            if not isinstance(captions, dict):
                raise ValueError("Caption file must contain a JSON object")
            
            self.logger.info(f"Loaded {len(captions)} captions")
            return captions
            
        except Exception as e:
            self.logger.error(f"Error loading caption file: {e}")
            raise
    
    def _find_valid_videos(self):
        """Find valid video files that have captions"""
        valid_files = []
        
        for video_name in self.captions.keys():
            video_path = self.data_dir / video_name
            if video_path.exists() and video_path.is_file():
                valid_files.append(video_name)
            else:
                self.logger.warning(f"Video file not found: {video_path}")
        
        return valid_files
    
    def _validate_dataset(self):
        """Validate dataset integrity"""
        self.logger.info("Validating dataset...")
        
        issues = []
        for video_name in self.video_files[:5]:  # Check first 5 videos
            try:
                video_path = self.data_dir / video_name
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    issues.append(f"Cannot open {video_name}")
                    continue
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                if frame_count == 0:
                    issues.append(f"{video_name} has no frames")
                
                if fps <= 0:
                    issues.append(f"{video_name} has invalid FPS")
                
                cap.release()
                
            except Exception as e:
                issues.append(f"Error validating {video_name}: {e}")
        
        if issues:
            self.logger.warning(f"Found {len(issues)} validation issues")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("Dataset validation passed")
    
    def __len__(self):
        return len(self.video_files)
    
    def load_video_robust(self, video_path):
        """Robust video loading with error handling"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Read all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError(f"No frames loaded from: {video_path}")
            
            # Sample frames to match target length
            if len(frames) > self.video_length:
                # Uniform sampling
                indices = np.linspace(0, len(frames)-1, self.video_length, dtype=int)
                frames = [frames[i] for i in indices]
            elif len(frames) < self.video_length:
                # Repeat frames to reach target length
                while len(frames) < self.video_length:
                    frames.extend(frames[:min(len(frames), self.video_length - len(frames))])
                frames = frames[:self.video_length]
            
            # Apply transforms
            video_tensor = torch.stack([self.transform(frame) for frame in frames])
            return video_tensor  # Shape: [T, C, H, W]
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {e}")
            # Return a dummy video tensor
            dummy_frame = Image.new('RGB', (self.resolution, self.resolution), color='black')
            video_tensor = torch.stack([self.transform(dummy_frame) for _ in range(self.video_length)])
            return video_tensor
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = self.data_dir / video_name
        caption = self.captions[video_name]
        
        # Load video
        video = self.load_video_robust(video_path)
        
        return {
            'video': video,
            'caption': caption,
            'video_name': video_name
        }

class EnhancedTextToVideoTrainer:
    """Enhanced trainer with monitoring and error handling"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.setup_directories()
        
        # Initialize models
        self.setup_models()
        
        # Setup dataset and dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
        
        # Setup monitoring
        self.setup_monitoring()
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.args.output_dir,
            os.path.join(self.args.output_dir, "logs"),
            os.path.join(self.args.output_dir, "plots"),
            os.path.join(self.args.output_dir, "samples"),
            os.path.join(self.args.output_dir, "checkpoints")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_models(self):
        """Initialize models with error handling"""
        try:
            self.logger.info("Loading models...")
            
            # Load tokenizer and text encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
            
            # Freeze text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # Load UNet
            self.unet = UNet3DConditionModel.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                subfolder="unet",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.unet.to(self.device)
            
            # Load scheduler
            self.scheduler = DDPMScheduler.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                subfolder="scheduler"
            )
            
            # Enable optimizations
            if hasattr(self.unet, 'enable_gradient_checkpointing'):
                self.unet.enable_gradient_checkpointing()
            
            # Enable xformers if available
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                self.logger.info("xformers attention enabled")
            except:
                self.logger.warning("xformers not available")
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def setup_data(self):
        """Setup dataset and dataloader"""
        try:
            self.dataset = EnhancedVideoDataset(
                data_dir=self.args.data_dir,
                caption_file=self.args.caption_file,
                video_length=self.args.video_length,
                resolution=self.args.resolution,
                validate=True
            )
            
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            self.logger.info(f"Dataset size: {len(self.dataset)}")
            self.logger.info(f"Batches per epoch: {len(self.dataloader)}")
            
        except Exception as e:
            self.logger.error(f"Error setting up data: {e}")
            raise
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Only train the UNet
        trainable_params = list(self.unet.parameters())
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        num_training_steps = len(self.dataloader) * self.args.epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"Total training steps: {num_training_steps}")
    
    def setup_monitoring(self):
        """Setup training monitoring"""
        self.metrics = {
            'step': [],
            'loss': [],
            'learning_rate': [],
            'epoch': [],
            'gpu_memory': [],
            'cpu_percent': []
        }
        
        self.best_loss = float('inf')
        self.log_file = os.path.join(self.args.output_dir, "logs", "training_metrics.csv")
        
        # Write CSV header
        with open(self.log_file, 'w') as f:
            f.write("step,epoch,loss,learning_rate,gpu_memory,cpu_percent,timestamp\n")
    
    def encode_text(self, captions):
        """Encode text captions with error handling"""
        try:
            inputs = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(**inputs).last_hidden_state
            
            return text_embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            # Return dummy embeddings
            batch_size = len(captions)
            return torch.zeros(batch_size, 77, 768, device=self.device)
    
    def train_step(self, batch):
        """Single training step with error handling"""
        try:
            videos = batch['video'].to(self.device, non_blocking=True)  # [B, T, C, H, W]
            captions = batch['caption']
            
            batch_size = videos.shape[0]
            
            # Encode text
            text_embeddings = self.encode_text(captions)
            
            # Rearrange video tensor for UNet input [B, C, T, H, W]
            videos = videos.permute(0, 2, 1, 3, 4)
            
            # Sample noise
            noise = torch.randn_like(videos)
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()
            
            # Add noise to videos
            noisy_videos = self.scheduler.add_noise(videos, noise, timesteps)
            
            # Predict noise
            model_pred = self.unet(
                noisy_videos,
                timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            
            # Calculate loss
            loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def get_system_stats(self):
        """Get system statistics"""
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        cpu_percent = psutil.cpu_percent()
        
        return gpu_memory, cpu_percent
    
    def log_metrics(self, step, epoch, loss, lr):
        """Log training metrics"""
        gpu_memory, cpu_percent = self.get_system_stats()
        
        self.metrics['step'].append(step)
        self.metrics['epoch'].append(epoch)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        self.metrics['gpu_memory'].append(gpu_memory)
        self.metrics['cpu_percent'].append(cpu_percent)
        
        # Write to CSV
        with open(self.log_file, 'a') as f:
            f.write(f"{step},{epoch},{loss:.6f},{lr:.2e},{gpu_memory:.2f},{cpu_percent:.1f},{datetime.now()}\n")
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.save_checkpoint(f"best_step_{step}")
    
    def plot_metrics(self):
        """Create and save training plots"""
        if len(self.metrics['step']) < 2:
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            ax1.plot(self.metrics['step'], self.metrics['loss'], 'b-', linewidth=2, alpha=0.7)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Learning rate plot
            ax2.plot(self.metrics['step'], self.metrics['learning_rate'], 'r-', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # GPU Memory plot
            if self.metrics['gpu_memory']:
                ax3.plot(self.metrics['step'], self.metrics['gpu_memory'], 'g-', linewidth=2, alpha=0.7)
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('GPU Memory (GB)')
                ax3.set_title('GPU Memory Usage')
                ax3.grid(True, alpha=0.3)
            
            # CPU Usage plot
            ax4.plot(self.metrics['step'], self.metrics['cpu_percent'], 'm-', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('CPU Usage (%)')
            ax4.set_title('CPU Usage')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.args.output_dir, "plots", "training_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
        try:
            checkpoint_dir = os.path.join(self.args.output_dir, "checkpoints", name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save UNet
            self.unet.save_pretrained(checkpoint_dir)
            
            # Save training state
            training_state = {
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'metrics': self.metrics,
                'best_loss': self.best_loss,
                'args': vars(self.args)
            }
            
            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            self.logger.info(f"Checkpoint saved: {name}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            # Load UNet
            self.unet = UNet3DConditionModel.from_pretrained(checkpoint_path)
            self.unet.to(self.device)
            
            # Load training state
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                self.optimizer.load_state_dict(training_state['optimizer'])
                self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
                self.metrics = training_state['metrics']
                self.best_loss = training_state['best_loss']
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
    
    def generate_sample(self, prompt, save_path=None):
        """Generate a sample video"""
        try:
            self.unet.eval()
            
            with torch.no_grad():
                # Encode text
                text_embeddings = self.encode_text([prompt])
                
                # Generate random noise
                noise = torch.randn(
                    1, 3, self.args.video_length, self.args.resolution, self.args.resolution,
                    device=self.device
                )
                
                # Denoise using scheduler
                for t in tqdm(self.scheduler.timesteps, desc="Generating video"):
                    timestep = t.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        model_pred = self.unet(
                            noise,
                            timestep,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False
                        )[0]
                    
                    noise = self.scheduler.step(model_pred, t, noise).prev_sample
                
                # Convert to video
                video = (noise / 2 + 0.5).clamp(0, 1)
                video = video.permute(0, 2, 1, 3, 4)[0]  # [T, C, H, W]
                
                # Save video if path provided
                if save_path:
                    self.save_video(video, save_path)
                
                return video
                
        except Exception as e:
            self.logger.error(f"Error generating sample: {e}")
            return None
        finally:
            self.unet.train()
    
    def save_video(self, video_tensor, save_path):
        """Save video tensor to file"""
        try:
            # Convert tensor to numpy
            video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()  # [H, W, C, T]
            video_np = (video_np * 255).astype(np.uint8)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 24.0, (video_np.shape[1], video_np.shape[0]))
            
            # Write frames
            for i in range(video_np.shape[3]):
                frame = video_np[:, :, :, i]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            out.release()
            self.logger.info(f"Video saved: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving video: {e}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        global_step = 0
        start_time = time.time()
        
        try:
            for epoch in range(self.args.epochs):
                self.unet.train()
                epoch_loss = 0.0
                
                progress_bar = tqdm(
                    self.dataloader, 
                    desc=f"Epoch {epoch+1}/{self.args.epochs}",
                    leave=False
                )
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Training step
                    self.optimizer.zero_grad()
                    
                    loss = self.train_step(batch)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    
                    # Update metrics
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    
                    # Log metrics
                    if global_step % self.args.log_interval == 0:
                        self.log_metrics(global_step, epoch, loss_value, current_lr)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss_value:.6f}",
                        'lr': f"{current_lr:.2e}",
                        'best_loss': f"{self.best_loss:.6f}"
                    })
                    
                    global_step += 1
                    
                    # Save checkpoint
                    if global_step % self.args.checkpoint_interval == 0:
                        self.save_checkpoint(f"step_{global_step}")
                    
                    # Generate sample
                    if global_step % self.args.sample_interval == 0:
                        sample_prompt = "A beautiful sunset over the ocean"
                        sample_path = os.path.join(
                            self.args.output_dir, "samples", f"sample_step_{global_step}.mp4"
                        )
                        self.generate_sample(sample_prompt, sample_path)
                    
                    # Plot metrics
                    if global_step % self.args.plot_interval == 0:
                        self.plot_metrics()
                    
                    # Memory cleanup
                    if global_step % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # End of epoch
                avg_loss = epoch_loss / len(self.dataloader)
                elapsed_time = time.time() - start_time
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.args.epochs} completed. "
                    f"Average loss: {avg_loss:.6f}, "
                    f"Time elapsed: {elapsed_time/3600:.2f}h"
                )
                
                # Save epoch checkpoint
                self.save_checkpoint(f"epoch_{epoch+1}")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            # Final cleanup
            self.save_checkpoint("final")
            self.plot_metrics()
            self.logger.info("Training completed")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Text-to-Video Fine-tuning')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--caption_file', type=str, required=True, help='JSON file with video captions')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Model arguments
    parser.add_argument('--video_length', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--resolution', type=int, default=512, help='Video resolution')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10, help='Steps between logging')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Steps between checkpoints')
    parser.add_argument('--sample_interval', type=int, default=500, help='Steps between samples')
    parser.add_argument('--plot_interval', type=int, default=100, help='Steps between plots')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, "logs"))
    
    # Log arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Create trainer
    trainer = EnhancedTextToVideoTrainer(args)
    
    # Load checkpoint if specified
    if args.checkpoint_path:
        trainer.load_checkpoint(args.checkpoint_path)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()