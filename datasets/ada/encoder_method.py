import os
import torch
import numpy as np
from PIL import Image
from trainers.addition import encoder as Encoder
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(size=(224,224)),
    T.CenterCrop((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
])

class EncoderM():
    def __init__(self,cfg):
        self.cfg = cfg
        self.__encoder_dir = ''  # cfg.TRAIN.DATA_INIT_ENCODER.DIR
        self.__dataset_name = '' # cfg.DATASET.NAME
        self.__encoder_name = 'raw' # cfg.TRAIN.DATA_INIT_ENCODER.NAME
        self.__encoder_type = ''  # cfg.TRAIN.DATA_INIT_ENCODER.FILE_TYPE
        self.__source = '-'.join(sorted(cfg.DATASET.SOURCE_DOMAINS))

        self.__encoder_pth = self.__gene_encoder_pth()  # get encoder path

        # Set device
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load encoder
        if self.__encoder_pth is not None:
            encoder = Encoder.__dict__[self.__encoder_name](**self.cfg.TRAIN.DATA_INIT_ENCODER.PARAMS)
            self.encoder = self.__load_encoder(encoder)
        else:
            self.encoder = None

    def __load_encoder(self, encoder):
        if self.__encoder_name.lower() in ['vqvae2']:
            checkpoint = torch.load(self.__encoder_pth, map_location=self.device)
            encoder.load_state_dict(checkpoint['state_dict'])
        else:
            raise "Please input right encoder's name!"
        
        return encoder
        
    def __gene_encoder_pth(self):
        if not self.__encoder_dir:
            return None
        if not self.__encoder_name:
            return None
        if not self.__encoder_type:
            return None
        path =os.path.join(self.__encoder_dir, self.__dataset_name, '{}-{}.{}'.format(self.__encoder_name, self.__source, self.__encoder_type))

        if os.path.exists(path):
            return path
        return None

    def __raw_dcenter(self,impaths):
        imgs = []
        for path in impaths:
            imgs.append(np.array(Image.open(path).resize((112, 112))).reshape(1, -1))
        imgs = np.concatenate(imgs, axis=0)
        return imgs
    
    @torch.no_grad()
    def __vqvae2_dcenter(self, impaths):
        encoder_imgs = []
        temp_imgs = []
        for i,path in enumerate(impaths):
            temp_imgs.append(transform(Image.open(path)))
            if i % 200 == 0:
                batch_imgs = torch.stack(temp_imgs, dim=0)
                embedding1, embedding2 = self.encoder(batch_imgs)
                embedding = torch.cat([
                        embedding1.view((embedding1.size(0), -1)),
                        embedding2.view((embedding2.size(0), -1))
                    ], dim=1)

                encoder_imgs.append(embedding.numpy())
                temp_imgs = []
        return np.concatenate(encoder_imgs, axis=0)
    
    def encoderf(self):
        if self.__encoder_name.lower() in ['vqvae2']:
            return self.__vqvae2_dcenter
        elif self.__encoder_name.lower() in ['raw']:
            return self.__raw_dcenter
        else:
            raise "Please input the right encoder's name!"