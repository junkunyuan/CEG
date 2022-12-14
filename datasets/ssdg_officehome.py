import os.path as osp
import glob
import random
from typing import Dict, List, Tuple

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum
from dassl.utils import mkdir_if_missing
from datasets.ada.database import ADA_DatasetBase

@DATASET_REGISTRY.register()
class SSDGOfficeHome(ADA_DatasetBase):
    """Office-Home.

    Statistics:
        - 4 domains: Art, Clipart, Product, Real world.
        - 65 categories.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
        - Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """
    dataset_dir = 'data-officehome'
    domains = ['art', 'clipart', 'product', 'real_world']
    data_url = 'https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa'

    def __init__(self, cfg):
        self.cfg = cfg
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_ssdg_dir = osp.join(self.dataset_dir, 'splits_ssdg')
        mkdir_if_missing(self.split_ssdg_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'office_home_dg.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        seed = cfg.SEED
        num_labeled = cfg.DATASET.NUM_LABELED/2
        src_domains = cfg.DATASET.SOURCE_DOMAINS
        tgt_domain = cfg.DATASET.TARGET_DOMAINS[0]
        split_ssdg_path = osp.join(self.split_ssdg_dir, f'{tgt_domain}_nlab{num_labeled}_seed{seed}_{self._split_pofix()}.json')

        if not osp.exists(split_ssdg_path):
            train_x, train_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS,num_labeled,cfg.TRAIN.DATA_MODE)
            self._write_json_train(split_ssdg_path, src_domains, self.dataset_dir, train_x, train_u)
        else:
            train_x, train_u = self._read_json_train(split_ssdg_path, src_domains, self.dataset_dir)
        val = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, 'val')
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, 'all')

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x
  
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test,num_labeled=num_labeled,cfg=cfg)
        self._num_classes = 65

    def load_data_train(self, input_domains: List[str], mode: str='all') -> Tuple[Dict or List, int, int]:
        """
        mode:
            all: return a list object containing all samples
            domain: return a dict containing samples of each domain
            domain_class: return a dict containing samples of each domain and each class
        """
        num_samples = 0
        num_class = None
        all_items = [] if mode == 'all' else {}
        
        for domain,dname in enumerate(input_domains):
            if mode in ['domain', 'domain_class']:
                all_items[dname] = [] if mode =='domain' else {}

            path = osp.join(self.dataset_dir, dname, "train")
            folders = listdir_nohidden(path, sort=True)
            num_class = len(folders)
            for label, folder in enumerate(folders):
                if mode == 'domain_class':
                    all_items[dname][label] = []

                impaths = glob.glob(osp.join(path, folder, '*.jpg'))
                random.shuffle(impaths)
                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain)
                    num_samples += 1
                    if mode =='domain_class':
                        all_items[dname][label].append(item)
                    elif mode == 'domain':
                        all_items[dname].append(item)
                    elif mode=='all':
                        all_items.append(item)

        return all_items, num_class, num_samples

    def _read_data_test(self, input_domains, split):

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory, sort=True)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, '*.jpg'))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            if split == 'all':
                train_dir = osp.join(self.dataset_dir, dname, 'train')
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, 'val')
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items
