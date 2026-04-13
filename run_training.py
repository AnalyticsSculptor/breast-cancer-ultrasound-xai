import yaml
import torch
from src.data.transforms import get_transforms
from src.data.dataset import prepare_data_loaders
from src.models.classifier import BreastCancerClassifier
from src.training.trainer import ModelTrainer

def main():
    # 1. Load Configuration
    print("Loading configuration...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Prepare Data
    print("Preparing data loaders...")
    transforms = {
        'train': get_transforms('train', config['data']['image_size']),
        'val': get_transforms('val', config['data']['image_size'])
    }

    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir=config['data']['raw_dir'],
        batch_size=config['data']['batch_size'],
        splits=config['data']['splits'],
        transforms=transforms
    )

    # 3. Initialize Model
    print("Initializing EfficientNet-B4 Model...")
    model = BreastCancerClassifier(
        backbone_name=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )

    # 4. Start Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.fit()
    print("Training complete! Check the outputs/checkpoints folder.")

if __name__ == "__main__":
    main()