from collections.abc import Callable
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


def transform(size: int) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
        ]
    )


def sample_image(size: int) -> torch.Tensor:
    """A deterministic synthetic image for tests and quick runs; no download needed."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    x = torch.stack(
        [
            0.25
            + 0.55
            * (xx > -0.35).float()
            * (xx < 0.5).float()
            * (yy > -0.5).float()
            * (yy < 0.45).float(),
            0.2 + 0.55 * torch.exp(-8 * (xx**2 + yy**2)),
            0.2 + 0.45 * (torch.sin(11 * xx) * torch.sin(11 * yy) > 0).float(),
        ]
    ).clamp(0, 1)
    return x


def load_dataset(cfg) -> Dataset:
    if cfg.name == "sample":
        return TensorDataset(sample_image(cfg.image_size).unsqueeze(0))
    if cfg.name not in {"pets", "celeba"}:
        image_dir = Path(to_absolute_path(cfg.root))
        image_name = Path(cfg.name)
        if image_name.name != cfg.name or "\\" in cfg.name:
            raise ValueError(
                "Select an image by filename only; image paths must be inside data.root."
            )
        if image_name.suffix:
            if image_name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Named images must be .jpg, .jpeg, or .png files.")
            image_path = image_dir / image_name
        else:
            image_path = next(
                (
                    image_dir / f"{cfg.name}{extension}"
                    for extension in (".jpg", ".jpeg", ".png")
                    if (image_dir / f"{cfg.name}{extension}").is_file()
                ),
                image_dir / f"{cfg.name}.jpg",
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image '{cfg.name}' was not found in {image_dir}. "
                "Use a filename stem such as 'astronaut' or add the image to data.root."
            )
        with Image.open(image_path) as image:
            tensor = transform(cfg.image_size)(image)
        return TensorDataset(tensor.unsqueeze(0))
    if cfg.name == "pets":
        return datasets.OxfordIIITPet(
            root=to_absolute_path(cfg.root),
            split=cfg.split,
            target_types=[],
            transform=transform(cfg.image_size),
            download=cfg.download,
        )
    if cfg.name == "celeba":
        return datasets.CelebA(
            root=to_absolute_path(cfg.root),
            split=cfg.split,
            target_type="attr",
            transform=transform(cfg.image_size),
            download=cfg.download,
        )
    raise ValueError(f"Unknown dataset: {cfg.name}")


def image_from_item(item):
    return item[0] if isinstance(item, (tuple, list)) else item
