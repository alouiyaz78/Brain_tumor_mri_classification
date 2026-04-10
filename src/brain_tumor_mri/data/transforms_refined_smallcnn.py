from torchvision import transforms

def get_smallcnn_refined_train_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
    ])

def get_smallcnn_refined_eval_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])