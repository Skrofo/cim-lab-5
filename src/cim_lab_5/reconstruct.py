import torch
import torch.nn.functional as F


def gradients(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def regulariser(x, method: str):
    dy, dx = gradients(x)
    if method == "tikhonov":
        # First-order Tikhonov: quadratic penalty on image gradients.
        return dx.square().mean() + dy.square().mean()
    if method == "tv":
        dx = F.pad(dx, (0, 1))
        dy = F.pad(dy, (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-6).mean()
    raise ValueError(f"Unknown reconstruction method: {method}")


def reconstruct(y, A, AT, cfg):
    # We need to choose our starting point for x (using the adjoint)
    # You can also try a null image (all zeros)
    
    #raise NotImplementedError("Design choice required: does it matter?")
    x = AT(y).detach().clone().clamp(0, 1).requires_grad_(True)
    x = torch.zeros_like(x).requires_grad_(True)

    # Choose an optimiser and the variable we are optimising
    opt = torch.optim.Adam([x], lr=cfg.learning_rate)
    history = []
    for step in range(cfg.iterations):
        opt.zero_grad()
        # Calculate our losses
        data_loss = F.mse_loss(A(x), y)
        prior_loss = regulariser(x, cfg.method)
        loss = data_loss + cfg.lambda_ * prior_loss

        # get the gradient of the losses
        loss.backward()
        # take an optimiser step
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if step % 25 == 0 or step == cfg.iterations - 1:
            history.append(
                (step, float(data_loss.detach()), float(prior_loss.detach()))
            )
    return x.detach(), history
