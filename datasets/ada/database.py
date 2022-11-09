import os.path as osp
from typing import Dict, List, Tuple

from dassl.data.datasets import Datum,DatasetBase
from dassl.utils import read_json,write_json

from datasets.ada import load_mode as LMODE
from datasets.ada.encoder_method import EncoderM

class ADA_DatasetBase(DatasetBase):
    def __init__(self, train_x, train_u, val, test, num_labeled,cfg):
        # num_labeled = cfg.DATASET.NUM_LABELED - num_labeled
        self.num_labeled = LMODE.calculate_real_num_labeled(num_labeled, len(train_u) + len(train_x))
        self.cfg = cfg
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
    
    def _read_json_train(self, filepath, src_domains, image_dir):
        def _convert_to_datums(items):
            out = []
            for impath, label, dname in items:
                if dname not in src_domains:
                    continue
                domain = src_domains.index(dname)
                impath = osp.join(image_dir, impath)
                item = Datum(impath=impath, label=int(label), domain=domain)
                out.append(item)
            return out
        
        print(f'Reading split from "{filepath}"')
        split = read_json(filepath)
        train_x = _convert_to_datums(split['train_x'])
        train_u = _convert_to_datums(split['train_u'])

        return train_x, train_u
    
    def _write_json_train(self, filepath, src_domains, image_dir, train_x, train_u):
        def _convert_to_list(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                domain = item.domain
                dname = src_domains[domain]
                impath = impath.replace(image_dir, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, dname))
            return out
        
        train_x = _convert_to_list(train_x)
        train_u = _convert_to_list(train_u)
        output = {
            'train_x': train_x,
            'train_u': train_u
        }

        write_json(output, filepath)
        print(f'Saved the split to "{filepath}"')
    
    def __init_em(self):
        if not hasattr(self,'em'):
            self.em = EncoderM(self.cfg)
        
    def _read_data(self, input_domains, num_labeled, mode='no_random'):
        self.__init_em()

        all_items, num_classes, num_samples = self.load_data_train(input_domains, LMODE.MODE_MAP[mode])
        num_domains = len(input_domains)
        # num_labeled = self.calculate_real_num_labeled(num_labeled,num_samples)
        return LMODE.METHOD[mode](all_items,num_labeled,
                                    num_samples=num_samples,
                                    num_classes=num_classes,
                                    num_domains=num_domains,
                                    encoderf=self.em.encoderf()
                                )

    def _split_pofix(self):
        """
        Split data.
        """
        if self.cfg.TRAIN.DATA_MODE in ['no_random', 'random_class', 'random_all']:
            return self.cfg.TRAIN.DATA_MODE
        elif self.cfg.TRAIN.DATA_MODE in ['dcenter']:
            return self.cfg.TRAIN.DATA_INIT_ENCODER.NAME + '_' + self.cfg.TRAIN.DATA_MODE
        else:
            raise 'Please input the right encoder name!'


    def _add_item(self, impath, label, domain, name='train_x'):
        print("Add item:", impath, label, domain)
        item = Datum(impath=impath, label=int(label), domain=int(domain))
        getattr(self, name).append(item)

    def _add_items(self, items, name='train_x'):
        for item in items:
            self._add_item(*item, name=name)

    def _remove_item(self, impath, label, domain, name='train_x'):
        item = Datum(impath=impath, label=int(label), domain=int(domain))
        getattr(self, name).remove(item)

    def _remove_items(self, items, name='train_x'):
        for item in items:
            self._remove_item(*item, name=name)
    def _set_data(self, newdata, name):
        if name == 'train_u':
            self._train_u = newdata
        elif name == 'train_x':
            self._train_x = newdata
    def load_data_train(self, input_domains:List[str], mode:str='all') -> Tuple[Dict or List,int,int]:
        raise NotImplemented
