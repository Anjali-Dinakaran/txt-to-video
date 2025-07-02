import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class TrainingMonitor:
    """Monitor and visualize training progress"""
    
    def __init__(self, log_file="training.log"):
        self.log_file = log_file
        self.metrics = {
            'step': [],
            'loss': [],
            'learning_rate': [],
            'timestamp': []
        }
    
    def log_metrics(self, step, loss, lr):
        """Log training metrics"""
        self.metrics['step'].append(step)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        self.metrics['timestamp'].append(datetime.now())
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()},{step},{loss:.6f},{lr:.2e}\n")
    
    def plot_metrics(self, save_path="training_plots.png"):
        """Create training plots"""
        if not self.metrics['step']:
            logger.warning("No metrics to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss plot
        ax1.plot(self.metrics['step'], self.metrics['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(self.metrics['step'], self.metrics['learning_rate'], 'r-', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {save_path}")
    
    def get_summary(self):
        """Get training summary"""
        if not self.metrics['step']:
            return "No training data available"
        
        summary = f"""
Training Summary:
- Total steps: {self.metrics['step'][-1]}
- Final loss: {self.metrics['loss'][-1]:.6f}
- Best loss: {min(self.metrics['loss']):.6f}
- Average loss: {sum(self.metrics['loss'])/len(self.metrics['loss']):.6f}
- Final learning rate: {self.metrics['learning_rate'][-1]:.2e}
- Training duration: {self.metrics['timestamp'][-1] - self.metrics['timestamp'][0]}
"""
        return summary