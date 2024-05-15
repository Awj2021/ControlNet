import os
# os.environ['WANDB_DIR'] = '/vol/research/wenjieProject/projects/owns/ControlNet/wandb'
# os.environ['WANDB_ARTIFACT_DIR'] = '/vol/research/wenjieProject/projects/owns/ControlNet/wandb_artifact'
from share import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from quilt1m_dataset import MyDataset
from finetune_dataset import ChaoyangDataset
from datetime import datetime

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# write the code with function. and add the __main__ function.
def train():
    # TODO: Please finish the model saving functions.
    # Configs
    resume_path = './cldm/7h572t94/checkpoints/finetune_quilt.ckpt'# Use the pretrained model from the Quilt Dataset.
    batch_size = 12
    logger_freq = 100
    learning_rate = 1e-6
    sd_locked = True
    only_mid_control = False
    save_dir = './logs'
    # TODO: task name should add the machine name.
    machine_name = os.uname()[1]
    task_name = f'finetuning_chaoyang_{machine_name}'
    model_dir = os.path.join(save_dir, 'models', task_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_chaoyang_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = ChaoyangDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    # wandb_logger = WandbLogger(name='control_sd15_ini_finetuning_quilt1M', project='cldm')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_logger = WandbLogger(name=f'control_sd15_ini_{timestamp}', project=f'cldm_{task_name}')
    logger = ImageLogger(save_dir=save_dir, batch_frequency=logger_freq, logger=wandb_logger)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='control_sd15_ini_{task_name}-{epoch:02d}-{steps:07d}',
    )

    trainer = pl.Trainer(gpus=[1], precision=16, callbacks=[logger, checkpoint_callback], logger=wandb_logger, max_epochs=100)
    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    train()