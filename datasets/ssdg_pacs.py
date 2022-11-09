import os.path as osp
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from dassl.data.datasets import DATASET_REGISTRY, Datum
from dassl.utils import mkdir_if_missing
from datasets.ada.database import ADA_DatasetBase

@DATASET_REGISTRY.register()
class SSDGPACS(ADA_DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
        - Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """
    dataset_dir = 'data-pacs'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    data_url = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    # the following images contain errors and should be ignored
    _error_paths = ['sketch/dog/n02103406_4068-1.png']

    def __init__(self, cfg):
        """
        Generate train_x, train_u, val and test
        """
        self.cfg = cfg
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = self.dataset_dir
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        self.split_ssdg_dir = osp.join(self.dataset_dir, 'splits_ssdg')

        mkdir_if_missing(self.split_ssdg_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'pacs.zip')
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
            train_x, train_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS, num_labeled, cfg.TRAIN.DATA_MODE)
            self._write_json_train(split_ssdg_path, src_domains, self.image_dir, train_x, train_u)
        else:
            train_x, train_u = self._read_json_train(split_ssdg_path, src_domains, self.image_dir)
        val = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, 'all')

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test,num_labeled=num_labeled, cfg=cfg)

    def load_data_train(self, input_domains: List[str], mode: str='all') -> Tuple[Dict or List, int, int]:
        """
        mode:
            all: return a list object containing all samples
            domain: return a dict containing samples of each domain
            domain_class: return a dict containing samples of each domain and each class
        """
        num_samples = 0
        num_class = None
        all_items = [] if mode =='all' else {}
        

        for domain,dname in enumerate(input_domains):
            if mode in ['domain','domain_class']:
                all_items[dname] = [] if mode == 'domain' else {}

            file = osp.join(self.split_dir, dname + '_train_kfold.txt')
            impath_label_list = self._read_split_pacs(file)

            impath_label_dict = defaultdict(list)
            for impath,label in impath_label_list:
                impath_label_dict[label].append((impath,label))
            labels = list(impath_label_dict.keys())
            num_class = len(labels)

            for label in labels:
                if mode == 'domain_class':
                    all_items[dname][label] = []

                pairs = impath_label_dict[label]
                random.shuffle(pairs)

                for impath,label in pairs:
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
        items = []

        for domain, dname in enumerate(input_domains):
            if split == 'all':
                file_train = osp.join(self.split_dir, dname + '_train_kfold.txt')
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(self.split_dir, dname + '_crossval_kfold.txt')
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(self.split_dir, dname + '_' + split + '_kfold.txt')
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(' ')
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items
