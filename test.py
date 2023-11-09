import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models as Model
import AudioLoader.music.mss as Dataset
import hydra
from hydra.utils import to_absolute_path

import os


@hydra.main(config_path="config", config_name="exp_test", version_base=None)
def my_app(cfg):
    cfg.dataset.data_root = to_absolute_path(cfg.dataset.data_root)

    testset = getattr(Dataset, cfg.dataset.name)(**cfg.dataset.test)

    testloader = torch.utils.data.DataLoader(testset, **cfg.dataloader.test)

    model = getattr(Model, cfg.model.name).load_from_checkpoint(cfg.checkpoint_path)


    name = f"Test-{cfg.model.name}-{cfg.sr}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)      
    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)


    trainer.test(model, testloader)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    my_app()