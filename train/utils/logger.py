import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Sequence

import torch as th
from torchrl.record import WandbLogger


class MyWandbLogger(WandbLogger):
    def __init__(
        self,
        exp_name: Optional[str],
        save_dir: Optional[Union[Path, str]],
        id: Optional[str],
        project_name: str = "malp",
        offline: bool = False,
        config: Optional[dict] = None,
        **kwargs,
    ):
        if exp_name is None:
            time_formatted = datetime.now().strftime(
                "%Hh-%Mm-%Ss"
            )  # : did not work, so we use _ instead
            date_formatted = datetime.now().strftime("%Y-%m-%d")
            model_type = config.get("model")
            exp_name = date_formatted + "/" + model_type + "_" + time_formatted

        if save_dir is None:
            current_dir = Path(os.getcwd())
            save_dir = Path(current_dir, "results", exp_name)
            save_dir.mkdir(parents=True, exist_ok=False)  # notify me if it exists

        super().__init__(exp_name, offline, save_dir, id, project_name, **kwargs)

    @staticmethod
    def get_log_variables_from_checkpoint(
        checkpoint_path: Union[str, Path], project_name: str = "malp"
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        date_formatted = checkpoint_path.parent.parent.parent.name
        give_name = checkpoint_path.parent.parent.name

        exp_name = date_formatted + "/" + give_name
        save_dir = checkpoint_path.parent.parent
        id = MyWandbLogger.get_id_from_experience(exp_name, project_name)

        return exp_name, save_dir, id

    @staticmethod
    def get_id_from_experience(exp_name: str, project_name: str = "malp"):
        import wandb

        api = wandb.Api()
        runs = api.runs(project_name)
        # Run name you're looking for
        target_run_name = exp_name
        # Search through runs in the project
        id = None
        for run in runs:
            if run.name == target_run_name:
                print("Found run ID:", run.id)
                id = run.id
                break
        if id is None:
            # Silently fall back to creating a new run. The old code called input()
            # here which hangs in non-interactive contexts (CI, scheduled jobs).
            print(f"No matching run for '{exp_name}' in project '{project_name}'. "
                  "Creating a new W&B run.")
        return id

    def log_image(self, name: str, image: Union[th.Tensor, Sequence], **kwargs) -> None:
        """Logs an image or a sequence of images to wandb.

        Args:
            name (str): The name of the image.
            image (Tensor or list of images): The image or images to be logged.
            **kwargs: Other keyword arguments. By construction, log_image
                supports 'step' (integer indicating the step index) and other
                kwargs are passed as-is to the :obj:`experiment.log` method.
        """
        import wandb

        if isinstance(image, th.Tensor):
            image = wandb.Image(image.detach().cpu().numpy())
        if isinstance(image, list):
            images = [wandb.Image(img) for img in image]
        else:
            images = [wandb.Image(image)]

        step = kwargs.pop("step", None)
        extra_kwargs = {}
        if step is not None:
            extra_kwargs["trainer/step"] = step
        self.experiment.log(
            {name: images, **extra_kwargs},
        )

    def save_checkpoint(self, policy, critic, optimizer, step: int, legacy_format: bool = False,
                        curriculum_state: dict = None):
        """Write a checkpoint under the logger's save directory.

        The loader in ``models/model_loader.py`` accepts both schemas. Default is the
        verbose ``*_state_dict`` keys; pass ``legacy_format=True`` to emit the short-key
        format that older tooling reads. ``curriculum_state`` (if given) persists the
        curriculum level/progress so a resume restarts at the right level.
        """
        checkpoint_dir = Path(self.save_dir, "checkpoints")
        checkpoint_path = Path(checkpoint_dir, f"checkpoint_{step}.pth")
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        if legacy_format:
            payload = {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
        else:
            payload = {
                "policy_state_dict": policy.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            }
        if curriculum_state is not None:
            payload["curriculum"] = curriculum_state
        th.save(payload, str(checkpoint_path))

    def save_network_summary(self, policy, critic, env):
        from torch_geometric.nn import summary

        net_summary_dir = Path(self.save_dir, "networks")
        if not net_summary_dir.exists():
            net_summary_dir.mkdir(parents=True)
        table = summary(policy, env.reset())
        with open(str(Path(net_summary_dir, "actor.txt")), "w") as f:
            f.write(str(table))
        print(table)

        table = summary(critic, env.reset())
        with open(str(Path(net_summary_dir, "critic.txt")), "w") as f:
            f.write(str(table))
        print(table)
