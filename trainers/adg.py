import contextlib
import time
import datetime

import numpy as np
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import Datum
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param,MetricMeter, AverageMeter

from utils.strategy import (get_embedding, get_grad_embedding)
from utils import strategy
from utils.util import test_loader, cal_num_active, initcenter
from utils.strategy import assign_label
from operator import itemgetter

def u_loader(data,cfg,tfm_train,dataset_wrapper=None):
    sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
    batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
    n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
    n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS
    if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
        sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
        batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS
    
    loader_u = build_data_loader(
        cfg,
        sampler_type=sampler_type_,
        data_source=data,
        batch_size=batch_size_,
        n_domain=n_domain_,
        n_ins=n_ins_,
        tfm=tfm_train,
        is_train=True,
        dataset_wrapper=dataset_wrapper
    )
    return loader_u
    

def kmeans(all_fea, all_label, accuracy, initc, discartc, sel_num, r=1.5, class_num=0, domain_num=3):
    pred_label,dd,K = assign_label(all_fea, initc, discartc, class_num, domain_num) 

    N = all_fea.shape[0]
    sel_num = N if sel_num > N else sel_num
                        

    select_sample = np.array([], dtype=np.int)
    sel_num_class = np.math.ceil(r * sel_num / (K - len(discartc)))
    for c in range(K):                  
        c_argsort = np.argsort(dd[:,c])
        select_sample = np.union1d(select_sample, c_argsort[:sel_num_class])
    
    np.random.shuffle(select_sample)
    index = select_sample[:sel_num]

    acc = np.sum(pred_label == all_label) / len(all_fea)
    log_str = 'After K-Means Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str)
    acc = np.sum(pred_label[index] == all_label[index]) / len(index)
    log_str = 'After K-Means Accuracy Select = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str)

    return index, pred_label


class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.linear(x)

class NormalClassifierD(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(num_features, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.linear3(x)

@TRAINER_REGISTRY.register()
class ADG(TrainerXU):

    def __init__(self, cfg):
        self.usedomain = cfg.TRAINER.ADG.USEDOMAIN
        super().__init__(cfg)

        self.alpha = cfg.TRAINER.ADG.ALPHA
        self.beta1,self.beta2 = cfg.TRAINER.ADG.BETA
        self.consistency = cfg.TRAINER.ADG.CONSISTENCY
        self.static = self.cfg.TRAINER.ADG.CLUSTER_STATIC
        self.cluster_p = self.cfg.TRAINER.ADG.CLUSTER_P
        self.cluster_init = self.cfg.TRAINER.ADG.CLUSTER_INIT/100

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.ADG.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == 'SeqDomainSampler'
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self,dataset=None,build_loader_u=True):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.ADG.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]

        self.custom_tfm_train = custom_tfm_train
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train, dataset=dataset)

        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        if build_loader_u:
            self.loader_u = u_loader(copy.deepcopy(self.dm.dataset.train_u), cfg, self.custom_tfm_train)
        self.num_classes = self.dm.num_classes

    
    def build_model(self):
        cfg = self.cfg

        print('Building G')
        self.G = SimpleNet(cfg, cfg.MODEL, 0)
        self.G.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model('G', self.G, self.optim_G, self.sched_G)
        
        print('Building Class C')
        self.CC = NormalClassifier(self.G.fdim, self.num_classes)
        self.CC.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.CC)))
        self.optim_CC = build_optimizer(self.CC, cfg.TRAINER.ADG.CC_OPTIM)
        self.sched_CC = build_lr_scheduler(self.optim_CC, cfg.TRAINER.ADG.CC_OPTIM)
        self.register_model('CC', self.CC, self.optim_CC, self.sched_CC)

        if self.usedomain:
            print('Building Domain C')
            self.DC = NormalClassifierD(self.G.fdim, self.dm.num_source_domains)
            self.DC.to(self.device)
            print('# params:  {:,}'.format(count_num_param(self.CC)))
            self.optim_DC = build_optimizer(self.DC, cfg.TRAINER.ADG.DC_OPTIM)
            self.sched_DC = build_lr_scheduler(self.optim_DC, cfg.TRAINER.ADG.DC_OPTIM)
            self.register_model('DC', self.DC, self.optim_DC, self.sched_DC)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()
        keep_rate = mask.sum() / mask.numel()
        output = {
            'acc_thre': acc_thre,
            'acc_raw': acc_raw,
            'keep_rate': keep_rate
        }
        return output

    def forward_backward(self, batch_x, batch_pu,batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_pu, batch_u)

        x = parsed_batch['x']
        x_aug = parsed_batch['x_aug']
        y_x_true = parsed_batch['y_x_true']

        u = parsed_batch['u']
        u_aug = parsed_batch['u_aug']
        y_u_true = parsed_batch['y_u_true']
        d_u_true = parsed_batch['d_u_true']
        loss_summary = {}

        K = self.dm.num_source_domains
        K = 2 if K == 1 else K
        
        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.CC(self.G(xu_k))
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= 0.95).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(y_xu_k_pred.chunk(2)[1])
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)
        
        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            f_x_k = self.G(x_k)
            z_x_k = self.CC(f_x_k)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)
        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.consistency:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.CC(f_xu_k_aug)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction='none')
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

        loss_all = 0
        if self.epoch + 1 >= self.cfg.TRAINER.ADG.CLUSTER_EPOCH and self.alpha > 0:
            pu_aug = parsed_batch['pu_aug']
            y_pu_clu = parsed_batch['y_pu_clu']
            loss_intra = self.intra_mixup(pu_aug,y_pu_clu,K)
            loss_inter = self.inter_mixup(pu_aug,y_pu_clu,K)
            # loss_mixup = self.beta*loss_intra+(1-self.beta)*loss_inter
            loss_mixup = self.beta1 * loss_intra + self.beta2 * loss_inter
            loss_summary['loss_intra_u'] = loss_intra.item()
            loss_summary['loss_inter_x'] = loss_inter.item()
            loss_summary['loss_mixup'] = loss_mixup.item()
            loss_summary['alpha_loss_mixup'] = (self.alpha * loss_mixup).item()
            loss_all += (self.alpha*loss_mixup)

        names = ['G','CC']
        loss_all += loss_x
        loss_summary['loss_x'] = loss_x.item()
        if self.consistency:
            loss_all += loss_u_aug
            loss_summary['loss_u_aug'] = loss_u_aug.item()
        
        self.model_backward_and_update(loss_all, names=names)
        
        if self.usedomain:
            count = d_acc_count = 0
            loss_d = 0
            for k in range(K):
                u_k = u[k]
                f_u_k = self.G(u_k)
                d_u_k = self.DC(f_u_k.detach())
                d_acc_count += torch.sum(torch.eq(torch.argmax(d_u_k, dim=1), d_u_true[k]))
                count += d_u_true[k].size(0)
                loss_d += F.cross_entropy(d_u_k, d_u_true[k])
            loss_summary['loss_d'] = loss_d.item()
            loss_summary['domain_acc'] = d_acc_count/count
            self.model_backward_and_update(loss_d, names=['DC'])

        loss_summary['y_u_pred_acc_thre'] = y_u_pred_stats['acc_thre']
        loss_summary['y_u_pred_acc_raw'] = y_u_pred_stats['acc_raw']
        loss_summary['y_u_pred_keep_rate'] = y_u_pred_stats['keep_rate']

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def inter_mixup(self, datas, labels, K, mix_alpha=0.2):
        loss = 0
        lams = np.random.beta(mix_alpha, mix_alpha, size=(K * K))

        for i in range(K):
            for j in range(i + 1, K):
                lam = lams[i * K + j]
                datas_ij = lam * datas[i] + (1 - lam) * datas[j]
                loss_a, loss_b = self.mix_loss(datas_ij, labels[i], labels[j])
                loss += (lam * loss_a + (1 - lam) * loss_b)
        return loss

    def intra_mixup(self, datas, labels, K, mix_alpha=0.2):
        loss = 0
        lams = np.random.beta(mix_alpha, mix_alpha, size=K)
        for k in range(K):
            datas_k = datas[k]
            labels_k = labels[k]
            lam = lams[k]
            index = torch.randperm(datas_k.size(0)).to(self.device)
            datas_k = lam  * datas_k + (1 - lam) * datas_k[index]

            loss_a, loss_b = self.mix_loss(datas_k, labels_k, labels_k[index])
            loss += (lam * loss_a + (1 - lam) * loss_b)
        return loss

    def mix_loss(self, datas, labels_a, labels_b):
        z = self.CC(self.G(datas))
        loss_a = F.cross_entropy(z, labels_a)
        loss_b = F.cross_entropy(z, labels_b)
        return loss_a,loss_b

    def parse_batch_train(self, batch_x,batch_pu,batch_u):

        # x0 = batch_x['img0'].to(self.device) # no augmentation
        x = batch_x['img'].to(self.device) # weak augmentation
        x_aug = batch_x['img2'].to(self.device) # strong augmentation
        y_x_true = batch_x['label'].to(self.device)

        # u0 = batch_u['img0'].to(self.device)
        u = batch_u['img'].to(self.device)
        u_aug = batch_u['img2'].to(self.device)
        y_u_true = batch_u['label'].to(self.device) 
        d_u_true = batch_u['domain'].to(self.device)

        pu = batch_pu['img'].to(self.device)
        pu_aug = batch_pu['img2'].to(self.device)
        y_pu_true = batch_pu['label'].to(self.device)

        K = self.dm.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        # x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        # u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)
        d_u_true = d_u_true.chunk(K)

        pu = pu.chunk(K)
        pu_aug = pu_aug.chunk(K)
        y_pu_true = y_pu_true.chunk(K)
        if self.epoch + 1 >= self.cfg.TRAINER.ADG.CLUSTER_EPOCH and self.alpha > 0:                       
            y_pu_clu = torch.tensor(itemgetter(*batch_pu['impath'])(self.selected_u), dtype=torch.long).to(self.device)        # clustering label
            y_pu_clu = y_pu_clu.chunk(K) 

    
        batch = {
            # x
            # 'x0': x0,
            'x': x,
            'x_aug': x_aug,
            'y_x_true': y_x_true,
            # u
            # 'u0': u0,
            'u': u,
            'u_aug': u_aug,
            'y_u_true': y_u_true,
            'd_u_true': d_u_true,
            # u
            'pu':pu,
            'pu_aug':pu_aug,
            'y_pu_true':y_pu_true,
        }
        if self.epoch + 1 >= self.cfg.TRAINER.ADG.CLUSTER_EPOCH and self.alpha > 0: 
            batch['y_pu_clu'] = y_pu_true

        return batch
    
    def model_inference(self, input):
        features = self.G(input)
        class_pred = self.CC(features)
        return class_pred

    def before_epoch(self):
        super().before_epoch()                
        if self.start_epoch != 0:
            self.al(self.start_epoch==0)

    def after_epoch(self):
        super().after_epoch()
        if self.start_epoch == 0:
            self.al(self.start_epoch==0)

    def al(self,isinit):

        if self.epoch == self.start_epoch:
            self.data_u = copy.deepcopy(self.dm.dataset.train_u)
            self.test_loader_u = test_loader(self.cfg, self.data_u)
            self.test_loader_x = test_loader(self.cfg, self.dm.dataset.train_x)
        
        print(f'Unlabeled Data Num: %d; Labeled Data Num: %d; DM Unlabeled Data Num: %d' % (len(self.data_u), len(self.dm.dataset.train_x), len(self.dm.dataset.train_u)))

        if self.epoch + 1 == self.cfg.TRAINER.ADG.SAVE_EPOCH:
            self.save_model(self.epoch, self.output_dir)
        
        if not self.cfg.TRAINER.UPDATE_EPOCHS or len(self.cfg.TRAINER.UPDATE_EPOCHS) == 0:
            update_epochs = [i for i in range(self.cfg.TRAINER.ADG.SAVE_EPOCH, self.max_epoch-1)]
        else:
            update_epochs = self.cfg.TRAINER.UPDATE_EPOCHS

        if self.cfg.TRAINER.ADG.UPDATE_DATA and self.epoch in update_epochs:
            dataset = self.dm.dataset
            num_active = cal_num_active(self.epoch, update_epochs, dataset.num_labeled)
            seleced_data = self.select_data(num_active, self.test_loader_u, self.test_loader_x)

            for impath, label, domain in zip(*seleced_data):
                item = Datum(impath, int(label.item()), int(domain.item()))
                self.data_u.remove(item)
                dataset._add_item(impath, label, domain, name='train_x')

            self.test_loader_u = test_loader(self.cfg, self.data_u)
            self.test_loader_x = test_loader(self.cfg, self.dm.dataset.train_x)
        
        if self.epoch + 1 + int(isinit) >= self.cfg.TRAINER.ADG.CLUSTER_EPOCH and self.alpha > 0:
            sel_num = int((len(self.dm.dataset.train_x) + len(self.data_u)) * self.cluster_p / 100)
            if self.static is False:
                sel_num = self.cluster_init * sel_num + (1 - self.cluster_init) * sel_num * ((self.epoch + 1) / self.max_epoch)
                sel_num = round(sel_num)
            selected_u, ulabel_dtum = self.kmeans(self.test_loader_u, self.test_loader_x, sel_num)
            self.selected_u = selected_u
            self.dm.dataset._set_data(ulabel_dtum, 'train_u')

        self.build_data_loader(self.dm.dataset, build_loader_u=False)
        self.loader_u = u_loader(self.data_u, self.cfg, self.custom_tfm_train)
        print(f'FixMatch Num: %d, Pseudo Num: %d, Labeled Num: %d'%(len(self.loader_u.dataset), len(self.train_loader_u.dataset), len(self.train_loader_x.dataset)))

    def after_train(self):
        print('Finished training')

        if not self.cfg.TEST.NO_TEST:
            self.test()

        self.save_model(self.epoch, self.output_dir)

        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        self.close_writer()

    def select_data(self, num_active, test_loader_u, test_loader_x):
        self.set_model_mode(mode='eval')
        strategy_name = self.cfg.TRAINER.ADG.ALMETHOD
        if strategy_name.lower() in ['coreset', 'margin', 'confidence', 'entropy', 'erm', 'bvsb', 'badge', 'ours', 'ours2', 'ours3']:
            if strategy_name == 'bvsb':
                strategy_name = 'margin'
            args = {
                'model':{'G': self.G, 'CC': self.CC, 'DC': self.DC},
                'device': self.device,
                'target_classes': self.num_classes,
                'loss': F.cross_entropy,
                'usedomain': self.usedomain
            }
            if strategy_name == 'badge':
                unrets = get_grad_embedding(args, test_loader_u)
            else: 
                unrets = get_embedding(args, test_loader_u)
            if strategy_name == 'coreset':
                args['usedomain'] = False
                larets = get_embedding(args, test_loader_x)
                indx = strategy.__dict__[strategy_name](unrets['embeddings'], num_active, device=self.device,labeled_embeddings=larets['embeddings'])
            elif strategy_name == 'badge':
                indx = strategy.__dict__[strategy_name](unrets['embeddings'], num_active, device=self.device)
            elif strategy_name in ['margin', 'erm', 'confidence', 'entropy']:
                indx = strategy.__dict__[strategy_name](unrets['probs'], num_active, device=self.device)
            elif strategy_name.startswith('ours'):
                kwags = {
                    'domains': unrets['domains'],
                    'probs': unrets['probs'],
                    'class_num': self.num_classes,
                    'domains_num': self.dm.num_source_domains,
                    'initcs': None,
                    'discartcs': None,
                    'diversity': self.cfg.TRAINER.ADG.DIVERSITY,
                    'gamma': self.cfg.TRAINER.ADG.GAMMA,
                    'uncertainty': strategy.__dict__[self.cfg.TRAINER.ADG.UNCERTAINTY],
                    'domainess': strategy.__dict__[self.cfg.TRAINER.ADG.DOMAINESS],
                    'domainess_flip': self.cfg.TRAINER.ADG.DOMAINESS_FLIP
                }
                if self.usedomain:
                    kwags['probs_d'] = unrets['probs_d']
                
                if strategy_name == 'ours':
                    initcs,discartcs = initcenter(self.dm.num_source_domains, self.device, test_loader_x, self.G, self.num_classes)
                    kwags['initcs'] = initcs
                    kwags['discartcs'] = discartcs
                    indx = strategy.__dict__[strategy_name](unrets['embeddings'], num_active, self.device, **kwags)
                elif strategy_name == 'ours2':
                    initcs,discartcs = initcenter(self.dm.num_source_domains, self.device, test_loader_x, self.G, self.num_classes, cat=True)
                    kwags['initcs'] = initcs
                    kwags['discartcs'] = discartcs
                    indx = strategy.__dict__[strategy_name](unrets['embeddings'], num_active, self.device, **kwags)
                elif strategy_name == 'ours3':
                    initcs,discartcs = initcenter(self.dm.num_source_domains, self.device, test_loader_x, self.G, self.num_classes, cat=True)
                    kwags['initcs'] = initcs
                    kwags['discartcs'] = discartcs
                    indx = strategy.__dict__[strategy_name](unrets['embeddings'], num_active, self.device, **kwags)

            return unrets['impaths'][indx], unrets['labels'][indx], unrets['domains'][indx]

    def kmeans(self, test_loader_u, test_loader_x, sel_num):
        self.set_model_mode(mode='eval')
        print('{} {} {}'.format('*' * 50, 'Start K-Means...', '*' * 50))
        args = {
                'model': {'G':self.G,'CC':self.CC},
                'device': self.device,
                'target_classes': self.num_classes,
                'loss': F.cross_entropy,
                'usedomain': False
        }
        unrets = get_embedding(args,test_loader_u)
        all_fea, all_domains, all_pth, all_label, all_output = unrets['embeddings'], unrets['domains'], unrets['impaths'], unrets['labels'], unrets['probs']
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        all_fea = all_fea.float().cpu()
        all_label = all_label.int().cpu().numpy()

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        # sel_num = int((len(self.dm.dataset.train_x)+len(self.data_u))*self.cfg.TRAINER.ADG.CLUSTER_P/100)
        print((len(self.dm.dataset.train_x) + len(self.data_u)), self.cfg.TRAINER.ADG.CLUSTER_P / 100, sel_num)
        r = self.cfg.TRAINER.ADG.LARGER_R

        initc = []
        discartc = []
        initc, discartc = initcenter(len(self.cfg.DATASET.SOURCE_DOMAINS), self.device, test_loader_x, self.G, self.num_classes, cat=True)
        all_fea = all_fea.numpy()
        index,pred_label = kmeans(all_fea, all_label, accuracy, initc, discartc, sel_num, r, class_num=self.num_classes, domain_num=self.dm.num_source_domains)
        ulabel = dict(zip(all_pth[index],pred_label[index]))
        ulabel_dtum = [Datum(all_pth[i], int(pred_label[i]), int(all_domains[i])) for i in index]
        print('{} {} {}'.format('*' * 50, 'Done!', '*' * 50))

        return ulabel,ulabel_dtum

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == 'train_x':
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == 'train_u':
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == 'smaller_one':
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError
        self.num_batches = len(self.loader_u)  

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)
        loader_u_iter = iter(self.loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_pu = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_pu = next(train_loader_u_iter)
            
            try:
                batch_u = next(loader_u_iter)
            except StopIteration:
                loader_u_iter = iter(self.loader_u)
                batch_u = next(loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_pu, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
