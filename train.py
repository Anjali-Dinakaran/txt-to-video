# Text-to-Video Fine-tuning Project
# Complete setup for fine-tuning Stable Video Diffusion with custom dataset

import os
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
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableVideoDiffusionPipeline, UNet3DConditionModel
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Custom dataset for video-text pairs"""
    
    def __init__(self, data_dir, caption_file, video_length=16, resolution=512, fps=24):
        self.data_dir = Path(data_dir)
        self.video_length = video_length
        self.resolution = resolution
        self.fps = fps
        
        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        
        self.video_files = []
        for video_name in self.captions.keys():
            video_path = self.data_dir / video_name
            if video_path.exists():
                self.video_files.append(video_name)
        
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Video transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.video_files)
    
    def load_video(self, video_path):
        """Load video frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        cap.release()
        
        # Sample frames to match video_length
        if len(frames) > self.video_length:
            indices = np.linspace(0, len(frames)-1, self.video_length, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.video_length:
            # Repeat last frame to reach video_length
            while len(frames) < self.video_length:
                frames.append(frames[-1])
        
        # Apply transforms
        video_tensor = torch.stack([self.transform(frame) for frame in frames])
        return video_tensor  # Shape: [T, C, H, W]
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = self.data_dir / video_name
        caption = self.captions[video_name]
        
        # Load video
        video = self.load_video(video_path)
        
        return {
            'video': video,
            'caption': caption,
            'video_name': video_name
        }

class TextToVideoTrainer:
    """Main trainer class for text-to-video fine-tuning"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.setup_models()
        
        # Setup dataset and dataloader
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_training()
    
    def setup_models(self):
        """Initialize the models"""
        logger.info("Loading models...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        
        # Load UNet (we'll use a simplified 3D UNet for this example)
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
        
        # Enable gradient checkpointing for memory efficiency
        self.unet.enable_gradient_checkpointing()
        
        logger.info("Models loaded successfully")
    
    def setup_data(self):
        """Setup dataset and dataloader"""
        self.dataset = VideoDataset(
            data_dir=self.args.data_dir,
            caption_file=self.args.caption_file,
            video_length=self.args.video_length,
            resolution=self.args.resolution
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Only train the UNet
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        num_training_steps = len(self.dataloader) * self.args.epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def encode_text(self, captions):
        """Encode text captions"""
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
    
    def train_step(self, batch):
        """Single training step"""
        videos = batch['video'].to(self.device)  # [B, T, C, H, W]
        captions = batch['caption']
        
        # Encode text
        text_embeddings = self.encode_text(captions)
        
        # Rearrange video tensor for UNet input [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4)
        
        # Sample noise
        noise = torch.randn_like(videos)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (videos.shape[0],), device=self.device
        ).long()
        
        # Add noise to videos
        noisy_videos = self.scheduler.add_noise(videos, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_videos,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        self.unet.train()
        global_step = 0
        
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            
            for batch in progress_bar:
                # Forward pass
                loss = self.train_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save final model
        self.save_checkpoint("final")
        logger.info("Training completed!")
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UNet
        self.unet.save_pretrained(checkpoint_dir / "unet")
        
        # Save training state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'step': step
        }, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

class VideoGenerator:
    """Generate videos from trained model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load the fine-tuned pipeline
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        
        # Load fine-tuned UNet
        if os.path.exists(model_path):
            self.pipeline.unet = UNet3DConditionModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
        
        self.pipeline.to(self.device)
        logger.info("Video generator initialized")
    
    def generate_video(self, prompt, num_frames=16, output_path="generated_video.mp4"):
        """Generate video from text prompt"""
        logger.info(f"Generating video for prompt: {prompt}")
        
        # Create a dummy first frame (for img2vid pipeline)
        first_frame = Image.new('RGB', (512, 512), color='black')
        
        # Generate video
        with torch.no_grad():
            frames = self.pipeline(
                image=first_frame,
                decode_chunk_size=8,
                num_frames=num_frames,
                height=512,
                width=512
            ).frames[0]
        
        # Save video
        self.save_video(frames, output_path)
        logger.info(f"Video saved to {output_path}")
    
    def save_video(self, frames, output_path, fps=24):
        """Save frames as video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
        
        for frame in frames:
            frame_np = np.array(frame)
            frame_cv2 = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_cv2)
        
        out.release()

def create_sample_dataset():
    """Create sample dataset structure"""
    # Create directories
    os.makedirs("data/videos", exist_ok=True)
    
    # Create sample captions file
    sample_captions = {
        "video1.mp4": "A cat playing with a ball in the garden",
        "video2.mp4": "A person walking in the rain",
        "video3.mp4": "Sunset over the ocean with waves",
        # Add more captions for your 50 videos
    }
    
    with open("data/captions.json", "w") as f:
        json.dump(sample_captions, f, indent=2)
    
    logger.info("Sample dataset structure created")

def main():
    parser = argparse.ArgumentParser(description="Text-to-Video Fine-tuning")
    parser.add_argument("--mode", choices=["train", "generate", "setup"], default="train")
    parser.add_argument("--data_dir", default="data/videos")
    parser.add_argument("--caption_file", default="data/captions.json")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--model_path", default="./checkpoints/final/unet")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    
    # Generation arguments
    parser.add_argument("--prompt", default="A beautiful sunset over mountains")
    parser.add_argument("--output_video", default="generated_video.mp4")
    parser.add_argument("--num_frames", type=int, default=16)
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        create_sample_dataset()
    elif args.mode == "train":
        trainer = TextToVideoTrainer(args)
        trainer.train()
    elif args.mode == "generate":
        generator = VideoGenerator(args.model_path)
        generator.generate_video(args.prompt, args.num_frames, args.output_video)

if __name__ == "__main__":
    main()