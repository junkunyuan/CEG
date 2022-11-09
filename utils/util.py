import numpy as np
from scipy.spatial.distance import cdist
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
import torch

def arg_nearest_center(features, numbers=-5, metric='cosine'):
    features = np.array(features)

    length = features.shape[0]
    numbers = (-numbers * length) // 100 if numbers < 0 else numbers

    center = np.mean(features, axis=0).reshape(1,-1)
    dd = cdist(features, center, metric).reshape(-1)
    indexes = np.argsort(dd)

    return indexes[:numbers], indexes[numbers:]


def test_loader(cfg, data):
    tfm_test = build_transform(cfg, is_train=False)
    batch_size = 100

    test_loader = build_data_loader(
        cfg,
        sampler_type='SequentialSampler',
        data_source=data,
        batch_size=batch_size,
        tfm=tfm_test,
        is_train=False,
        dataset_wrapper=None
    )
    return test_loader


def cal_num_active(epoch, update_epochs, num_labeled):
    every_num = np.math.ceil(num_labeled / len(update_epochs))
    if epoch == update_epochs[-1]:
        num_active = num_labeled - (every_num) * (len(update_epochs) - 1)
    else:
        num_active = every_num
    return num_active


def initcenter(domain_num, device, loader, encoder, num_classes, cat=False):
    encoder.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data['img'].to(device)
            labels = data['label']
            domains = data['domain']
            feas = encoder(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_domains = domains.int()
                all_label = labels.int()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_domains = torch.cat((all_domains, domains.int()),0)
                all_label = torch.cat((all_label, labels.int()), 0)
    all_domains = all_domains.numpy()
    all_label = all_label.numpy()
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    initc = []
    discartc = []
    for domain in range(domain_num):
        d_index = np.where(all_domains==domain)[0]
        d_all_fea = all_fea[d_index]
        d_all_label = all_label[d_index]
        d_all_label = np.eye(num_classes)[d_all_label]

        d_initc = d_all_label.transpose().dot(d_all_fea)
        d_initc = d_initc / (1e-8 + d_all_label.sum(axis=0)[:, None])
        d_discartc = np.where(np.sum(d_initc,axis=1)==0)[0]

        if len(d_discartc) > 0:
            d_discartc += domain * num_classes 
        initc.append(d_initc)
        discartc.append(d_discartc)

    if cat == True:
        initc = np.concatenate(initc, axis=0)
        discartc = np.concatenate(discartc, axis=0)
    
    return initc, discartc