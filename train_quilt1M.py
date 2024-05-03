import os
# os.environ['WANDB_DIR'] = '/vol/research/wenjieProject/projects/owns/ControlNet/wandb'
# os.environ['WANDB_ARTIFACT_DIR'] = '/vol/research/wenjieProject/projects/owns/ControlNet/wandb_artifact'
from share import *
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from quilt1m_dataset import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# write the code with function. and add the __main__ function.
def train():

# Configs
    resume_path = './models/control_sd15_ini.ckpt'# firstly run the copy file.
    batch_size = 2
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    save_dir = './logs'


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_quilt1m_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(save_dir=save_dir, batch_frequency=logger_freq)
    wandb_logger = WandbLogger(name='control_sd15_ini_finetuning_quilt1M', project='cldm')

    trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], logger=wandb_logger, max_epochs=100)
    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    train()