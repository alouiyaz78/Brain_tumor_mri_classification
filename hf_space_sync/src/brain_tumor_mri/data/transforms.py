from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FixedHorizontalFlip:
    def __call__(self, img):
        return transforms.functional.hflip(img)


class FixedRotate:
    def __init__(self, angle: float):
        self.angle = angle

    def __call__(self, img):
        return transforms.functional.rotate(img, self.angle)


def get_train_transforms(
    img_size: int = 224,
    augmentation_level: str = "standard",
) -> transforms.Compose:
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ]

    if augmentation_level == "advanced":
        transform_list.extend([
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(img_size: int = 224):
    return [
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            FixedHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            FixedRotate(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            FixedRotate(-5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]