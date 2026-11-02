import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/MPPR-Net.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='VisDrone2019.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=8,
                device='0',
                # resume='',
                project='runs/train',
                name='exp',
                )