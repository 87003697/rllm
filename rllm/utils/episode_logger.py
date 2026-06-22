"""Episode JSON Logger for saving detailed episode information."""

import hashlib
import json
from pathlib import Path
from typing import Any

from rllm.eval.episode_store import _json_default, _sanitize
from rllm.types import Episode

# Training payload fields to exclude (save disk, visualizer doesn't need them)
_STEP_EXCLUDE = {"prompt_ids", "response_ids", "logprobs", "routing_matrices", "mc_return", "advantage", "weight_version", "model_output"}


class EpisodeLogger:
    """Logger to save episodes to individual JSON files with step and data hash."""

    def __init__(self, base_dir: str, subdirectory: str = "episodes"):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Kept for backward compatibility, ignored.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        task_str = json.dumps(task, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the run directory for a specific training step.

        Each step becomes a "run" directory with an episodes/ subdirectory,
        matching the layout that rllm view expects.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the run directory
        """
        run_dir = self.base_dir / f"{mode}_step_{step}_epoch_{epoch}"
        episodes_dir = run_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")
        filename = f"episode_hash{task_hash}_id{episode_id_safe}.json"
        return filename

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        # Exclude training payload arrays (prompt_ids, logprobs, etc.) from every step
        data = episode.model_dump(
            mode="json",
            exclude={
                "trajectories": {
                    "__all__": {          # every trajectory
                        "steps": {
                            "__all__": _STEP_EXCLUDE,  # every step
                        },
                    },
                },
            },
        )
        data["training_step"] = step
        data["training_epoch"] = epoch

        run_dir = self.get_step_dir(step, mode, epoch)
        filename = self.get_episode_filename(episode, step)
        filepath = run_dir / "episodes" / filename

        try:
            with open(filepath, "w") as f:
                json_str = json.dumps(data, indent=2, default=_json_default)
                f.write(json_str + "\n")
                f.flush()
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                print(f"[EpisodeLogger] Successfully logged episode {i + 1}/{len(episodes)}: {episode.id}")
            except Exception as e:
                print(f"[EpisodeLogger] Failed to log episode {i + 1}/{len(episodes)}: {e}")
                raise

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a meta.json in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a meta.json file for the batch
        """
        self.log_episodes(episodes, step, mode, epoch)

        if batch_summary and episodes:
            run_dir = self.get_step_dir(step, mode, epoch)
            meta = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "n_episodes": len(episodes),
                "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes),
            }
            with open(run_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
