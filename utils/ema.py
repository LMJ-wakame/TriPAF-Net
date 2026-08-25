def update_ema(ema_model, model, alpha):
    """Update ema_model parameters: ema = alpha * ema + (1-alpha) * model"""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)


def clone_model(model):
    """Return a deep-copied model with detached weights for EMA teacher init"""
    import copy

    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    return m
