import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import lpips

class VideoEvaluator:
    """Evaluate generated videos"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load LPIPS model for perceptual similarity
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        except:
            logger.warning("LPIPS not available, skipping perceptual metrics")
            self.lpips_model = None
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images (simplified version)"""
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def evaluate_video_quality(self, generated_frames, reference_frames=None):
        """Evaluate video quality metrics"""
        metrics = {}
        
        if reference_frames is not None:
            # Calculate metrics against reference
            psnr_scores = []
            ssim_scores = []
            lpips_scores = []
            
            for gen_frame, ref_frame in zip(generated_frames, reference_frames):
                # Convert to tensors if needed
                if not isinstance(gen_frame, torch.Tensor):
                    gen_frame = TF.to_tensor(gen_frame).unsqueeze(0).to(self.device)
                if not isinstance(ref_frame, torch.Tensor):
                    ref_frame = TF.to_tensor(ref_frame).unsqueeze(0).to(self.device)
                
                # Calculate PSNR and SSIM
                psnr = self.calculate_psnr(gen_frame, ref_frame)
                ssim = self.calculate_ssim(gen_frame, ref_frame)
                
                psnr_scores.append(psnr.item())
                ssim_scores.append(ssim.item())
                
                # Calculate LPIPS if available
                if self.lpips_model is not None:
                    lpips_score = self.lpips_model(gen_frame * 2 - 1, ref_frame * 2 - 1)
                    lpips_scores.append(lpips_score.item())
            
            metrics['psnr'] = sum(psnr_scores) / len(psnr_scores)
            metrics['ssim'] = sum(ssim_scores) / len(ssim_scores)
            
            if lpips_scores:
                metrics['lpips'] = sum(lpips_scores) / len(lpips_scores)
        
        # Calculate temporal consistency
        if len(generated_frames) > 1:
            temporal_consistency = []
            for i in range(len(generated_frames) - 1):
                frame1 = generated_frames[i]
                frame2 = generated_frames[i + 1]
                
                if not isinstance(frame1, torch.Tensor):
                    frame1 = TF.to_tensor(frame1).unsqueeze(0).to(self.device)
                if not isinstance(frame2, torch.Tensor):
                    frame2 = TF.to_tensor(frame2).unsqueeze(0).to(self.device)
                
                # Calculate frame difference
                diff = F.l1_loss(frame1, frame2)
                temporal_consistency.append(diff.item())
            
            metrics['temporal_consistency'] = sum(temporal_consistency) / len(temporal_consistency)
        
        return metrics