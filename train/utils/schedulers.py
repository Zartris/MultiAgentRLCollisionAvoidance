import torch as th


class EntropyDecay:
    def __init__(
        self,
        device,
        initial_entropy_coef,
        final_entropy_coef,
        decay_steps,
        decay_start_step=0,
        loss_module=None,
    ):
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.decay_steps = decay_steps
        self.loss_module = loss_module
        self.decay_start_step = decay_start_step
        self.device = device

    def get_value(self, step):
        return self.get_entropy_coef(step)

    def get_entropy_coef(self, step):
        if step < self.decay_start_step:
            return self.initial_entropy_coef
        if step >= self.decay_start_step + self.decay_steps:
            return self.final_entropy_coef
        else:
            decaying_steps = step - self.decay_start_step
            return self.initial_entropy_coef - (
                self.initial_entropy_coef - self.final_entropy_coef
            ) * (decaying_steps / self.decay_steps)
            # 1 - 1 * (20 / 100) = 1

    def step(self, step):
        new_entropy_coef = self.get_entropy_coef(step)
        if self.loss_module is not None:
            value = th.tensor(new_entropy_coef, device=self.device)
            # torchrl >=0.11 renamed the loss buffer entropy_coef -> entropy_coeff;
            # the repo's own ClipPPOLoss (train/utils/PPOLoss.py) still uses
            # entropy_coef. Update whichever this loss module actually exposes.
            if hasattr(self.loss_module, "entropy_coeff"):
                self.loss_module.entropy_coeff = value
            else:
                self.loss_module.entropy_coef = value
        return new_entropy_coef


class LRDecay:
    def __init__(self, optimizer, lr_start, lr_end, decay_steps, decay_start_step=0):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step
        # Snapshot each group's base LR so decay scales them PROPORTIONALLY. The
        # optimizer may carry an asymmetric actor/critic split (e.g. actor 2e-5,
        # critic 4e-4); the old step() overwrote every group to the single decayed
        # actor LR, silently destroying the critic's separate (much larger) LR.
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]

    def get_value(self, step):
        return self.get_lr(step)

    def get_lr(self, step):
        if step < self.decay_start_step:
            return self.lr_start
        if step >= self.decay_start_step + self.decay_steps:
            return self.lr_end

        decaying_steps = step - self.decay_start_step
        return self.lr_start - (self.lr_start - self.lr_end) * (
            decaying_steps / self.decay_steps
        )

    def step(self, step):
        new_lr = self.get_lr(step)
        # Scale every group by the SAME factor relative to its own base LR, preserving
        # any actor/critic ratio instead of flattening all groups to one LR.
        factor = (new_lr / self.lr_start) if self.lr_start else 1.0
        for base_lr, param_group in zip(self._base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * factor
        return new_lr


class STDDecay:
    """Linearly anneal the policy's std_max between `start` and `end`.

    Caller is expected to pass the *inner* module that actually owns `set_std_max`
    (typically a LocalNavigation* net). Previously this constructor walked a
    hard-coded `model.module[0].module[0]` chain to reach it; that assumption broke
    any time the wrapper structure changed. Callers now unwrap explicitly — see
    `model_loader.py` / `LidarSingleStep.py` for the one-line accessor.
    """

    def __init__(self, model, start, end, decay_steps, decay_start_step=0):
        if not hasattr(model, "set_std_max"):
            # Back-compat: if the caller passed the full wrapped tree, try the old
            # nesting so existing scripts keep working. Log once so the next refactor
            # knows it's still being used.
            try:
                model = model.module[0].module[0]
            except (AttributeError, IndexError, TypeError) as exc:
                raise AttributeError(
                    "STDDecay expected a module with `set_std_max`; got "
                    f"{type(model).__name__}. Unwrap the actor net manually before "
                    "passing it in."
                ) from exc
        self.model = model
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.decay_start_step = decay_start_step

    def get_value(self, step):
        return self.get_std_max(step)

    def get_std_max(self, step):
        if step < self.decay_start_step:
            return self.start
        if step >= self.decay_start_step + self.decay_steps:
            return self.end

        decaying_steps = step - self.decay_start_step
        return self.start - (self.start - self.end) * (
            decaying_steps / self.decay_steps
        )

    def step(self, step):
        std_max = self.get_std_max(step)
        self.model.set_std_max(std_max)
        return std_max
