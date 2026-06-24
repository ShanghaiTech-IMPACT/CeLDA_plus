import math


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        initial_lr,
        min_lr=0,
        warmup_start_lr=0,
        steps_per_epoch=1,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.steps_per_epoch = steps_per_epoch
        self.warmup_iters = warmup_epochs * steps_per_epoch
        self.max_iters = max_epochs * steps_per_epoch
        self.current_iter = 0

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            alpha = self.current_iter / max(1, self.warmup_iters)
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * alpha

        progress = (self.current_iter - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_mult" in param_group:
                param_group["lr"] = lr * param_group["lr_mult"]
            else:
                param_group["lr"] = lr
        self.current_iter += 1
        return lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class WarmupPolyScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        initial_lr,
        power=0.9,
        min_lr=0,
        warmup_start_lr=0,
        steps_per_epoch=1,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.power = power
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.steps_per_epoch = steps_per_epoch
        self.warmup_iters = warmup_epochs * steps_per_epoch
        self.max_iters = max_epochs * steps_per_epoch
        self.current_iter = 0

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            alpha = self.current_iter / max(1, self.warmup_iters)
            return self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * alpha

        progress = (self.current_iter - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
        poly_decay = (1 - progress) ** self.power
        return self.min_lr + (self.initial_lr - self.min_lr) * poly_decay

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_mult" in param_group:
                param_group["lr"] = lr * param_group["lr_mult"]
            else:
                param_group["lr"] = lr
        self.current_iter += 1
        return lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def get_scheduler(optimizer, scheduler_type="cosine", **kwargs):
    if scheduler_type in {"cosine", "cosine_warmup"}:
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            max_epochs=kwargs.get("max_epochs", 50),
            initial_lr=kwargs.get("initial_lr", 0.01),
            min_lr=kwargs.get("min_lr", 1e-6),
            warmup_start_lr=kwargs.get("warmup_start_lr", 0),
            steps_per_epoch=kwargs.get("steps_per_epoch", 1),
        )
    if scheduler_type in {"poly", "poly_warmup"}:
        return WarmupPolyScheduler(
            optimizer=optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            max_epochs=kwargs.get("max_epochs", 50),
            initial_lr=kwargs.get("initial_lr", 0.01),
            power=kwargs.get("power", 0.9),
            min_lr=kwargs.get("min_lr", 1e-6),
            warmup_start_lr=kwargs.get("warmup_start_lr", 0),
            steps_per_epoch=kwargs.get("steps_per_epoch", 1),
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
