import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from .data import image_from_item, load_dataset
from .operators import make_operator
from .reconstruct import reconstruct


def device_from(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def highpass_noise(x, sigma):
    noise = torch.randn_like(x)
    size = 2 * int(3 * sigma) + 1
    low = F.avg_pool2d(noise, kernel_size=size, stride=1, padding=size // 2)
    hp = noise - low
    return hp / (hp.square().mean().sqrt() + 1e-8)


def nullspace_challenge(x, A, cfg):
    y = A(x)
    best = None
    for _ in range(cfg.trials):
        delta = highpass_noise(x, cfg.highpass_sigma)
        scale = cfg.perturbation_scale * random.random()
        candidate = (x + scale * delta).clamp(0, 1)
        image_change = F.mse_loss(candidate, x).sqrt()
        measurement_change = F.mse_loss(A(candidate), y).sqrt()
        if measurement_change <= cfg.measurement_tolerance and (
            best is None or image_change > best[0]
        ):
            best = (image_change, measurement_change, candidate)
    return best


def show_triplet(x, y, third, titles, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, im, title in zip(axes, [x, y, third], titles):
        a = im[0].detach().cpu()
        if a.shape[0] == 1:
            ax.imshow(a[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(a.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = device_from(cfg.device)
    out = Path.cwd()
    print(OmegaConf.to_yaml(cfg))
    print(f"Using {device}; outputs: {out}")
    ds = load_dataset(cfg.data)
    A, AT = make_operator(cfg.operator, cfg.data.image_size, device)

    if cfg.mode == "dataset":
        n = min(cfg.data.max_examples, len(ds))
        xs, ys = [], []
        for i in range(n):
            xi = image_from_item(ds[i]).unsqueeze(0).to(device)
            raise NotImplementedError("Task")
            measurement += cfg.operator.noise_sigma * torch.randn_like(measurement)
            xs.append(xi.cpu())
            raise NotImplementedError("Task")
            ys.append()

        # Export using torch.save
        torch.save(
            {"target": torch.cat(xs), "measurement": torch.cat(ys)},
            out / "training_pairs.pt",
        )
        print(f"Saved {n} target/measurement pairs to training_pairs.pt")
        return

    if cfg.mode not in {"forward", "nullspace", "reconstruct", "all"}:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    x = image_from_item(ds[cfg.data.index]).unsqueeze(0).to(device)
    y_clean = A(x)
    y = y_clean + cfg.operator.noise_sigma * torch.randn_like(y_clean)

    if cfg.mode in {"forward", "all"}:
        save_image(x, out / "target.png")
        save_image(y.clamp(0, 1), out / "measurement.png")
        display_y = AT(y).clamp(0, 1)
        show_triplet(
            x,
            display_y,
            (x - display_y).abs(),
            ["Target", "Measurement (adjoint view)", "Absolute difference"],
            out / "forward_model.png",
        )

    if cfg.mode in {"nullspace", "all"}:
        best = nullspace_challenge(x, A, cfg.nullspace)
        if best is None:
            print(
                "No candidate met the tolerance. Increase nullspace.measurement_tolerance."
            )
        else:
            di, dm, candidate = best
            measurement_difference = AT((A(candidate) - A(x)).abs()).clamp(0, 1)
            show_triplet(
                x,
                candidate,
                measurement_difference,
                [
                    "Original",
                    "Largest hidden change",
                    "Measurement change (adjoint view)",
                ],
                out / "nullspace.png",
            )
            print(
                f"Null-space challenge: image RMSE={di:.4f}, measurement RMSE={dm:.4f}"
            )

    if cfg.mode in {"reconstruct", "all"}:
        # OmegaConf reserves lambda in attribute syntax; copy to a friendly name.
        rcfg = OmegaConf.create(
            OmegaConf.to_container(cfg.reconstruction, resolve=True)
        )
        rcfg.lambda_ = rcfg["lambda"]
        xhat, history = reconstruct(y, A, AT, rcfg)
        show_triplet(
            x,
            AT(y).clamp(0, 1),
            xhat,
            ["Target", "Adjoint / naive view", rcfg.method.title()],
            out / "reconstruction.png",
        )
        print("step, data term, prior term")
        for row in history:
            print(row)


if __name__ == "__main__":
    main()
