import os
import torch
import mlflow
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score
from .loss import FocalLoss

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = FocalLoss(gamma=2.0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['weight_decay']
        )
        self.scaler = torch.cuda.amp.GradScaler() # For Mixed Precision (AMP)
        
        self.best_sensitivities = [] # To track top-3 checkpoints
        self.output_dir = "outputs/checkpoints"
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # AMP Context
            with torch.autocast(device_type=self.device):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient Clip
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate Sensitivity (Recall for the positive class - Malignant)
        sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        return sensitivity

    def manage_checkpoints(self, epoch, sensitivity):
        """Saves top-3 checkpoints based on validation sensitivity."""
        ckpt_path = os.path.join(self.output_dir, f"model_ep{epoch}_sens{sensitivity:.4f}.pth")
        
        self.best_sensitivities.append((sensitivity, ckpt_path))
        self.best_sensitivities.sort(key=lambda x: x[0], reverse=True)
        
        # Save current model
        torch.save(self.model.state_dict(), ckpt_path)
        
        # Keep only top 3, delete older ones
        if len(self.best_sensitivities) > 3:
            _, path_to_remove = self.best_sensitivities.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

    def fit(self):
        mlflow.start_run()
        mlflow.log_params(self.config['training'])
        
        # Phase 1: Frozen Backbone
        print("Starting Phase 1: Training Head Only...")
        self.model.freeze_backbone()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Phase 2: Unfreeze Backbone
            if epoch == self.config['training']['freeze_epochs'] + 1:
                print("Starting Phase 2: Full Fine-tuning...")
                self.model.unfreeze_backbone()
                
            train_loss = self.train_epoch()
            val_sensitivity = self.validate()
            
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Sensitivity: {val_sensitivity:.4f}")
            
            mlflow.log_metrics({"train_loss": train_loss, "val_sensitivity": val_sensitivity}, step=epoch)
            self.manage_checkpoints(epoch, val_sensitivity)
            
        mlflow.end_run()