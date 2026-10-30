import math

import deepinv as dinv
import torch


def motion_kernel(
    size: int, angle_degrees: float, device: torch.device
) -> torch.Tensor:
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    a = math.radians(angle_degrees)
    along = (xx - c) * math.cos(a) + (yy - c) * math.sin(a)
    across = -(xx - c) * math.sin(a) + (yy - c) * math.cos(a)
    k = ((along.abs() <= c) & (across.abs() <= 0.55)).float()
    return (k / k.sum()).view(1, 1, size, size)


def defocus_kernel(size: int, radius: float, device: torch.device) -> torch.Tensor:
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    k = (((xx - c) ** 2 + (yy - c) ** 2) <= radius**2).float()
    return (k / k.sum()).view(1, 1, size, size)


def make_bayer_operator(image_size: int, device: torch.device):
    """Pack DeepInverse's masked RGB output as a one-channel raw mosaic."""
    physics = dinv.physics.Demosaicing(
        img_size=(3, image_size, image_size), pattern="bayer", device=device
    )

    def forward(x):
        return physics.A(x).sum(dim=1, keepdim=True)

    def adjoint(y):
        masked_rgb = y.expand(-1, 3, -1, -1)
        return physics.A_adjoint(masked_rgb)

    return forward, adjoint


def make_operator(cfg, image_size: int, device: torch.device):
    name = cfg.name

    if name == "bayer":
        return make_bayer_operator(image_size, device)
    if name == "downsample":
        physics = dinv.physics.Downsampling(
            img_size=(3, image_size, image_size),
            filter=None,
            factor=cfg.downsample.factor,
            device=device,
        )
        return physics.A, physics.A_adjoint

    if name == "gaussian":
        kernel = dinv.physics.functional.gaussian_blur(
            psf_size=(cfg.gaussian.kernel_size, cfg.gaussian.kernel_size),
            sigma=(cfg.gaussian.sigma, cfg.gaussian.sigma),
            device=str(device),
        )
    elif name == "motion":
        kernel = motion_kernel(cfg.motion.kernel_size, cfg.motion.angle_degrees, device)
    elif name in {"defocus", "defocus_bayer"}:
        kernel = defocus_kernel(cfg.defocus.kernel_size, cfg.defocus.radius, device)
    else:
        raise ValueError(f"Unknown operator: {name}")

    blur = dinv.physics.Blur(filter=kernel, padding="circular", device=device)
    if name == "defocus_bayer":
        bayer, bayer_adjoint = make_bayer_operator(image_size, device)

        def forward(x):
            return bayer(blur.A(x))

        def adjoint(y):
            return blur.A_adjoint(bayer_adjoint(y))

        return forward, adjoint

    return blur.A, blur.A_adjoint
