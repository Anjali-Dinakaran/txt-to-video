class Config:
    """Configuration settings for text-to-video training"""
    
    # Model settings
    MODEL_NAME = "stabilityai/stable-video-diffusion-img2vid"
    TEXT_ENCODER = "openai/clip-vit-base-patch32"
    
    # Data settings
    VIDEO_LENGTH = 16
    RESOLUTION = 512
    FPS = 24
    
    # Training settings
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 100
    WARMUP_STEPS = 500
    SAVE_STEPS = 500
    MAX_GRAD_NORM = 1.0
    
    # Paths
    DATA_DIR = "data/videos"
    CAPTION_FILE = "data/captions.json"
    OUTPUT_DIR = "./checkpoints"
    
    # Hardware
    MIXED_PRECISION = True
    GRADIENT_CHECKPOINTING = True
    NUM_WORKERS = 2

# data_validator.py - Validate dataset
