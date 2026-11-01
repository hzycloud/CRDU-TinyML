from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_imagefolder(
    root: str | Path,
    image_size: int = 32,
    normalize: bool = False,
) -> ImageFolder:
    operations = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        operations.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )
    return ImageFolder(root=str(root), transform=transforms.Compose(operations))

