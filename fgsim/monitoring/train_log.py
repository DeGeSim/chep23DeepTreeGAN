from datetime import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.monitoring import logger


class TrainLog:
    """Initialized with the `holder`,
    provides the logging with wandb/tensorboard."""

    def __init__(self, state):
        # This code block is formatting the hyperparameters
        # for the experiment and creating a list of tags.
        self.state: DictConfig = state
        default_log = (
            conf.command in ["train", "test", "generate"] and not conf.debug
        )
        if conf.command in ["test", "generate"]:
            default_log = True
        self.use_tb = default_log
        self.use_wandb = default_log

        _epoch = self.state["epoch"]
        _step = self.state["grad_step"]

        self.log_cmd_time()

    def log_cmd_time(self):
        if self.use_wandb:
            dstr = datetime.now().strftime("%y-%m-%d-%H:%M")
            logger.info(f"time/{conf.command}" + dstr)


    def log_metrics(
        self,
        metrics_dict: dict[str, Union[float, torch.Tensor, np.float32]],
        step,
        epoch,
        prefix=None,
    ):
        if prefix is not None:
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}

        # self._log_metrics_tb(metrics_dict, step, epoch)
        # self._log_metrics_wandb(metrics_dict, step, epoch)

    def flush(self):
        pass
    #     if self.use_wandb:
    #         if len(self._wandb_tmp):
    #             logger.info("Wandb flush")
    #             if conf.command == "test":
    #                 self._flush_test_wandb()
    #             else:
    #                 wandb.log(
    #                     self._wandb_tmp
    #                     | {
    #                         "epoch": self._wandb_epoch,
    #                         "grad_step": self._wandb_step,
    #                     }
    #                 )
    #             self._wandb_tmp = {}
    #             self._wandb_step = None
    #             self._wandb_epoch = None

    
    def log_figure(
        self,
        figure_name: str,
        figure: Figure,
        step: int,
        epoch: int,
    ):
        if self.use_tb:
            self.writer.add_figure(tag=figure_name, figure=figure, global_step=step)
        if self.use_wandb:
            self._set_wandb_state(step, epoch)
            self._wandb_tmp.update({f"p/{figure_name}": wandb.Image(figure)})
        plt.close(figure)

    # def log_test_metrics(
    #     self,
    #     metrics_dict: dict[str, Union[float, torch.Tensor]],
    #     step: int,
    #     epoch: int,
    #     prefix: str,
    # ):
    #     metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
    #     self._log_metrics_tb(metrics_dict, step, epoch)
    #     self._log_metrics_wandb(metrics_dict, step, epoch)
    #     if self.use_wandb:
    #         for k, v in metrics_dict.items():
    #             self.wandb_run.summary[k] = v

    def write_trainstep_logs(self, interval) -> None:
        if not all(
            [
                hasattr(self.state, time)
                for time in [
                    "time_train_step_end",
                    "time_train_step_start",
                    "time_io_end",
                ]
            ]
        ):
            return
        traintime = (
            self.state.time_train_step_end - self.state.time_train_step_start
        )
        iotime = self.state.time_io_end - self.state.time_train_step_start
        utilisation = 1 - iotime / traintime

        self.log_metrics(
            {
                "batchtime": traintime / interval,
                "utilisation": utilisation / interval,
                "processed_events": self.state.processed_events / interval,
            },
            self.state["grad_step"],
            self.state["epoch"],
            prefix="speed",
        )

    def next_epoch(self) -> None:
        self.state["epoch"] += 1

        if self.use_tb:
            self.writer.add_scalar(
                "epoch",
                self.state["epoch"],
                self.state["grad_step"],
                new_style=True,
            )

    def __del__(self):
        if self.use_tb:
            self.writer.flush()
            self.writer.close()

    def end(self) -> None:
        pass
