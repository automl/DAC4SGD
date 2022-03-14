import os
from stable_baselines3.common.callbacks import CheckpointCallback


class CustomCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        verbose: int = 0,
        override: bool = False,
    ):
        """
        Callback for saving a model every ``save_freq`` calls
        to ``env.step()``.

        .. warning::

          When using multiple environments, each call to  ``env.step()``
          will effectively correspond to ``n_envs`` steps.
          To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

        :param save_freq:
        :param save_path: Path to the folder where the model will be saved.
        :param name_prefix: Common prefix to the saved models
        :param verbose:
        :param override: bool. If false, model file will get a unique identifier (number
            of timesteps) and is not overriden.
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            verbose=verbose,
        )
        self.override = override

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_id = f"{self.name_prefix}"
            if not self.override:
                model_id += f"_{self.num_timesteps}_steps"
            path = os.path.join(self.save_path, model_id)
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True
