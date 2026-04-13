import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(phase: str, image_size: int = 224):
    if phase == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        # Validation, Test, and Inference phase
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])