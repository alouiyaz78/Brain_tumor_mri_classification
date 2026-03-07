from torchvision import transforms

def get_train_transforms(img_size: int = 224, augmentation_level: str = 'standard') -> transforms.Compose:
    """
    Crée le pipeline de transformation pour l'entraînement.
    'standard' : Votre base actuelle (Rotation/Flip).
    'advanced' : Ajoute Flou, Contraste et Zoom pour raffiner les résultats.
    """
    transform_list = [
        transforms.Resize((img_size, img_size)), # Redimensionnement pour le modèle 
        transforms.RandomHorizontalFlip(p=0.5), # Diversité de position
        transforms.RandomRotation(degrees=10),   # Diversité d'inclinaison
    ]

    # Ajout du niveau 'advanced' pour le raffinage
    if augmentation_level == 'advanced':
        transform_list.extend([
            # Simule des variations de capteurs IRM
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            # Simule un léger flou de mouvement
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # Focus sur des parties de l'image pour les petites tumeurs
            transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.0))
        ])

    transform_list.extend([
        transforms.ToTensor(), # Conversion en tenseur mathématique 
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), # Standards ImageNet 
            std=(0.229, 0.224, 0.225),
        ),
    ])
    
    return transforms.Compose(transform_list)

def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Transformations pour validation/test : stricte conversion sans altération."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])