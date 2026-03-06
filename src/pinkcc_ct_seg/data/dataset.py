from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png"}


class BrainMRIDataset(Dataset):
    """
    Dataset binaire :
    - 0 = no_tumor
    - 1 = tumor (glioma / meningioma / pituitary)
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dossier introuvable : {self.root_dir}")

        for cls_dir in sorted(self.root_dir.iterdir()):
            if not cls_dir.is_dir():
                continue

            label = 0 if cls_dir.name == "no_tumor" else 1

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"Aucune image trouvée dans {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label