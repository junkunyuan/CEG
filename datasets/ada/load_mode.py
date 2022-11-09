import random
from utils.util import arg_nearest_center
import numpy as np

def calculate_real_num_labeled(old_num_labeled, num_samples):
    num_labeled = old_num_labeled
    if num_labeled < 0:  # if num_labeled is ratio
        assert num_labeled >= -100, 'When -100<= num_labeled < 0, it represents the percentage of samples to use.'
        num_labeled  = int((-num_labeled * num_samples) / 100)
    return num_labeled


def split_random_all(all_items, num_labeled, **kwags):
    num_labeled = calculate_real_num_labeled(num_labeled, len(all_items))
    random.shuffle(all_items)
    items_x = all_items[:num_labeled]
    items_u = all_items[num_labeled:]
    return items_x, items_u


def split_random_class(all_items, num_labeled, **kwags):
    items_x,items_u = [],[]
    for k, v in all_items.items():
        num_labeled_per_domain = calculate_real_num_labeled(num_labeled, len(v))
        random.shuffle(v)
        items_x += v[:num_labeled_per_domain]
        items_u += v[num_labeled_per_domain:]
    return items_x, items_u


def split_no_random(all_items,num_labeled, **kwags):
    items_x,items_u = [], []
    num_samples = kwags['num_samples']
    num_labeled = calculate_real_num_labeled(num_labeled,num_samples)

    num_labeled_per_class = num_labeled // (kwags['num_domains'] * kwags['num_classes'])

    for _, ddata in all_items.items():
        for _, cdata in ddata.items():
            items_x += cdata[:num_labeled_per_class]
            items_u += cdata[num_labeled_per_class:]
    return items_x, items_u

def split_dcenter(all_items, num_labeled, **kwags):
    encoderf = kwags['encoderf']
    items_x, items_u = [], []

    for _, items in all_items.items():
        num_labeled_per_domain = calculate_real_num_labeled(num_labeled, len(items))
        items = np.array(items)
        imgs = []
        for item in items:
            imgs.append(item.impath)
        imgs = encoderf(imgs)
        xindex, uindex = arg_nearest_center(imgs, num_labeled_per_domain)
        
        items_x.extend(list(items[xindex]))
        items_u.extend(list(items[uindex]))

    return items_x, items_u  


METHOD = {
    'random_class': split_random_class,
    'no_random': split_no_random,
    'random_all': split_random_all,
    'dcenter': split_dcenter
}

MODE_MAP = {
    'no_random': 'domain_class',
    'random_class': 'domain',
    'random_all': 'all',
    'dcenter': 'domain'
}
