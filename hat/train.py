# flake8: noqa
import os
import os.path as osp
import sys
sys.path.append('.')
os.environ['RANK'] = '0'
import hat.archs
import hat.data
import hat.models
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
