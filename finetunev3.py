#!/usr/bin/env python3
"""
Simplified Text-to-Video Fine-tuning Script
Compatible version that avoids diffusers library issues
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
import time
import psutil
import gc
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Import required libraries
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as TF
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install transformers torch torchvision")
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

class SimpleVideoUNet(nn.Module):
    """Simplified 3D UNet for video generation"""
    
    def __init__(self, in_channels=4, out_channels=4, text_embed_dim=512):
        super().__init__()
        
        # Time embedding
        self.time_embed_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )
        
        # Text embedding projection
        self.text_proj = nn.Linear(text_embed_dim, self.time_embed_dim)
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64, self.time_embed_dim)
        self.enc2 = self.conv_block(64, 128, self.time_embed_dim)
        self.enc3 = self.conv_block(128, 256, self.time_embed_dim)
        self.enc4 = self.conv_block(256, 512, self.time_embed_dim)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, self.time_embed_dim)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512, self.time_embed_dim)
        self.dec3 = self.conv_block(512 + 256, 256, self.time_embed_dim)
        self.dec2 = self.conv_block(256 + 128, 128, self.time_embed_dim)
        self.dec1 = self.conv_block(128 + 64, 64, self.time_embed_dim)
        
        # Output
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch, time_embed_dim):
        """3D Convolution block with time conditioning"""
        return nn.ModuleDict({
            'conv1': nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            'norm1': nn.GroupNorm(8, out_ch),
            'conv2': nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            'norm2': nn.GroupNorm(8, out_ch),
            'time_mlp': nn.Linear(time_embed_dim, out_ch),
            'activation': nn.GELU(),
        })
    
    def apply_conv_block(self, x, block, time_embed):
        """Apply a conv block with time conditioning"""
        # First conv
        h = block['conv1'](x)
        h = block['norm1'](h)
        h = block['activation'](h)
        
        # Add time conditioning
        time_h = block['time_mlp'](time_embed)
        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1, 1]
        time_h = time_h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + time_h
        
        # Second conv
        h = block['conv2'](h)
        h = block['norm2'](h)
        h = block['activation'](h)
        
        return h
    
    def get_time_embedding(self, timesteps):
        """Get sinusoidal time embeddings"""
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, timesteps, encoder_hidden_states=None):
        """Forward pass"""
        # Get time embeddings
        time_embed = self.get_time_embedding(timesteps)
        time_embed = self.time_mlp(time_embed)
        
        # Add text conditioning
        if encoder_hidden_states is not None:
            # Pool text embeddings
            text_embed = encoder_hidden_states.mean(dim=1)  # [B, 768]
            text_embed = self.text_proj(text_embed)  # [B, 256]
            time_embed = time_embed + text_embed
        
        # Encoder
        e1 = self.apply_conv_block(x, self.enc1, time_embed)
        e1_pool = nn.functional.max_pool3d(e1, 2)
        
        e2 = self.apply_conv_block(e1_pool, self.enc2, time_embed)
        e2_pool = nn.functional.max_pool3d(e2, 2)
        
        e3 = self.apply_conv_block(e2_pool, self.enc3, time_embed)
        e3_pool = nn.functional.max_pool3d(e3, 2)
        
        e4 = self.apply_conv_block(e3_pool, self.enc4, time_embed)
        e4_pool = nn.functional.max_pool3d(e4, 2)
        
        # Bottleneck
        b = self.apply_conv_block(e4_pool, self.bottleneck, time_embed)
        
        # Decoder
        d4 = nn.functional.interpolate(b, scale_factor=2, mode='trilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.apply_conv_block(d4, self.dec4, time_embed)
        
        d3 = nn.functional.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.apply_conv_block(d3, self.dec3, time_embed)
        
        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.apply_conv_block(d2, self.dec2, time_embed)
        
        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.apply_conv_block(d1, self.dec1, time_embed)
        
        # Output
        output = self.final_conv(d1)
        
        return output

class SimpleScheduler:
    """Simple DDPM scheduler"""
    
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Create beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
    
    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples"""
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

class VideoDataset(Dataset):
    """Simplified video dataset"""
    
    def __init__(self, data_dir, caption_file, video_length=16, resolution=256):
        self.data_dir = Path(data_dir)
        self.video_length = video_length
        self.resolution = resolution
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load captions
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        
        # Find valid videos
        self.video_files = []
        for video_name in self.captions.keys():
            video_path = self.data_dir / video_name
            if video_path.exists():
                self.video_files.append(video_name)
        
        self.logger.info(f"Found {len(self.video_files)} videos")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        
        # Sample frames
        if len(frames) > self.video_length:
            indices = np.linspace(0, len(frames)-1, self.video_length, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.video_length:
            while len(frames) < self.video_length:
                frames.extend(frames[:min(len(frames), self.video_length - len(frames))])
            frames = frames[:self.video_length]
        
        # Transform frames
        video_tensor = torch.stack([self.transform(frame) for frame in frames])
        return video_tensor  # [T, C, H, W]
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = self.data_dir / video_name
        caption = self.captions[video_name]
        
        try:
            video = self.load_video(video_path)
        except:
            # Dummy video if loading fails
            dummy_frame = Image.new('RGB', (self.resolution, self.resolution))
            video = torch.stack([self.transform(dummy_frame) for _ in range(self.video_length)])
        
        return {'video': video, 'caption': caption}

class SimpleTrainer:
    """Simplified trainer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
        
        # Setup models
        self.setup_models()
        
        # Setup data
        self.setup_data()
        
        # Setup training
        self.setup_training()
        
        # Metrics
        self.metrics = {'step': [], 'loss': [], 'lr': []}
        self.best_loss = float('inf')
    
    def setup_models(self):
        """Setup models"""
        try:
            # Text encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
            
            # Freeze text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # UNet
            self.unet = SimpleVideoUNet(
                in_channels=3,  # RGB
                out_channels=3,
                text_embed_dim=512
            )
            self.unet.to(self.device)
            
            # Scheduler
            self.scheduler = SimpleScheduler()
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def setup_data(self):
        """Setup dataset"""
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
    
    def setup_training(self):
        """Setup optimizer"""
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01
        )
        
        # Simple scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.dataloader) * self.args.epochs
        )
    
    def encode_text(self, captions):
        """Encode text"""
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
        """Training step"""
        videos = batch['video'].to(self.device)  # [B, T, C, H, W]
        captions = batch['caption']
        
        # Rearrange to [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4)
        
        # Encode text
        text_embeddings = self.encode_text(captions)
        
        # Add noise
        noise = torch.randn_like(videos)
        batch_size = videos.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # Add noise to videos
        noisy_videos = self.scheduler.add_noise(videos, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.unet(
            noisy_videos,
            timesteps,
            encoder_hidden_states=text_embeddings
        )
        
        # Calculate loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        ckpt_path = os.path.join(self.args.output_dir, "checkpoints", f"model_step_{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_loss': self.best_loss
        }, ckpt_path)
        self.logger.info(f"Checkpoint saved at step {step}")

    def train(self):
        """Training loop"""
        global_step = 0
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            epoch_loss = 0.0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch}"):
                self.unet.train()
                self.optimizer.zero_grad()
                
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                self.metrics['step'].append(global_step)
                self.metrics['loss'].append(loss.item())
                self.metrics['lr'].append(self.optimizer.param_groups[0]['lr'])

                if global_step % self.args.log_every == 0:
                    self.logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                # Save checkpoint
                if global_step % self.args.save_every == 0:
                    self.save_checkpoint(global_step)

                # Save best model
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    torch.save(self.unet.state_dict(), os.path.join(self.args.output_dir, "best_model.pt"))
                    self.logger.info(f"Best model saved at step {global_step} with loss {self.best_loss:.4f}")

            self.logger.info(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / len(self.dataloader):.4f}")

        # Final save
        self.save_checkpoint(global_step)
        self.logger.info("Training completed.")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Text-to-Video Fine-tuning Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to video data folder")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to JSON caption file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--video_length", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of video frames")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    trainer = SimpleTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
