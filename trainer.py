import argparse
from configs.config import load_config
# General config

#from model_quat import  train_manifold2 as train_manifold
from model import train_naive  as train_manifold
from model import load_data
import shutil
import ipdb
from data.data_splits import amass_splits
def train(opt,config_file):

    data_load =getattr(load_data, opt['experiment']['data_name'])
    train_dataset = data_load('train', data_path=opt['data']['data_dir'], amass_splits=amass_splits['train'],batch_size=opt['train']['batch_size'],num_workers=opt['train']['num_worker'])
    val_dataset = data_load('val', data_path=opt['data']['data_dir'], amass_splits=amass_splits['vald'],batch_size=opt['train']['batch_size'], num_workers=opt['train']['num_worker'])


    trainer = getattr(train_manifold, opt['experiment']['type'])
    trainer = trainer( train_dataset=train_dataset, val_dataset=val_dataset, opt=opt)

    copy_config = '{}/{}/{}'.format(opt['experiment']['root_dir'], trainer.exp_name, 'config.yaml')
    shutil.copyfile(config_file,copy_config )

    trainer.train_model(opt['train']['max_epoch'], eval=opt['train']['eval'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train nd manifold.'
    )
    parser.add_argument('--config', '-c', default='configs/amass.yaml', type=str, help='Path to config file.')
    args = parser.parse_args()

    opt = load_config(args.config)
    #save the config file

    train(opt, args.config)