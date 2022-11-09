import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import pdb

from scipy import stats
from scipy.spatial.distance import cdist

__all__=['get_embedding','get_grad_embedding',
        'corset','badge','margin','entropy','confidence','erm'
]

def parse_batch_test(batch,device):
    input = batch['img'].to(device)
    label = batch['label'].to(device)
    impath = batch['impath']
    domain = batch['domain']

    res = {
        'x': input,
        'y':label,
        'pth':impath,
        'd':domain
    }

    return res

def predict(models,batch,parse_batch_func,freeze=False,usedomain=False,**kwags):
    device = kwags['device']
    G = models['G']
    CC = models['CC']
    res = parse_batch_func(batch,device)
    pred = {}
    if freeze:
        with torch.no_grad():
            feature = G(res['x'])
    else:
        feature = G(res['x'])
    pred['fea'] = feature
    pred['CC'] = CC(feature)
    pred['prob'] = F.softmax(pred['CC'],dim=-1)
    if usedomain: 
        DC = models['DC']
        pred['DC'] =  DC(feature)
        pred['probs_d'] = F.softmax(pred['DC'],dim=-1)

    pred['impath'] = res['pth']
    pred['domain'] = res['d']
    pred['label'] = res['y']
    
    return pred

def get_embedding(args,u_loader):
    model = args['model']
    device = args['device']
    usedomain = args['usedomain']

    for k,mod in model.items():
        model[k] = mod.to(device)
        model[k].eval()
    
    embeddings = []
    domains = []
    impaths = []
    labels = []
    probs = []
    probs_d = []
        
    with torch.no_grad():
        for _, elements_to_predict in enumerate(u_loader):
            pred = predict(model, elements_to_predict, parse_batch_test, device=device, usedomain=usedomain)
            embeddings.append(pred['fea'])
            probs.append(pred['prob'])
            domains.append(pred['domain'])
            labels.append(pred['label'])
            impaths.append(pred['impath'])
            if usedomain:
                probs_d.append(pred['probs_d'])
    embeddings = torch.cat(embeddings, dim=0)
    probs = torch.cat(probs, dim=0)
    probs_d = torch.cat(probs_d, dim=0) if usedomain else None
    domains = torch.cat(domains, dim=0)
    labels = torch.cat(labels, dim=0)
    impaths = np.concatenate(impaths, axis=0)

    rets = {
        'embeddings': embeddings,
        'domains': domains,
        'impaths': impaths,
        'labels': labels,
        'probs': probs
    }
    if usedomain:
        rets['probs_d'] = probs_d
    return rets

def get_grad_embedding(args, u_loader, predict_labels=True, grad_embedding_type="bias_linear"):
    model = args['model']
    device = args['device']
    target_classes = args['target_classes']
    lossf= args['loss']
    usedomain = args['usedomain']

    for k, mod in model.items():
        model[k] = mod.to(device)

    embDim = model['G'].fdim
    
    grad_embeddings = []
    domains = []
    impaths = []
    labels = []
    probs = []
    probs_d = []
    
    for batch_idx, unlabeled_data_batch in enumerate(u_loader):
        pred = predict(model,unlabeled_data_batch,parse_batch_test,freeze=True,device=device,usedomain=usedomain)
        out,l1,targets = pred['CC'],pred['fea'],pred['label']
        probs.append(pred['prob'])
        domains.append(pred['domain'])
        impaths.append(pred['impath'])
        labels.append(pred['label'])
        if usedomain:
            probs_d.append(pred['prob_d'])
        if predict_labels:
            targets = out.max(1)[1]
        
        loss = lossf(out, targets, reduction="sum")
        l0_grads = torch.autograd.grad(loss, out)[0]

        if grad_embedding_type != "bias":
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, target_classes)

        if grad_embedding_type == "bias":                
            grad_embeddings.append(l0_grads)
        elif grad_embedding_type == "linear":
            grad_embeddings.append(l1_grads)
        else:
            grad_embeddings.append(torch.cat([l0_grads, l1_grads], dim=1))

        torch.cuda.empty_cache()

    grad_embeddings = torch.cat(grad_embeddings, dim=0)
    probs = torch.cat(probs, dim=0)
    probs_d = torch.cat(probs_d, dim=0)  if usedomain else None
    domains = torch.cat(domains, dim=0)
    labels = torch.cat(labels, dim=0)
    impaths = np.concatenate(impaths, axis=0)
    rets = {
        'embeddings': grad_embeddings,
        'domains': domains,
        'impaths': impaths,
        'labels': labels,
        'probs': probs,
    }
    if usedomain:
        rets['probs_d'] = probs_d
    return rets

def coreset(unlabeled_embeddings, n, device, **kwags):
    labeled_embeddings = kwags['labeled_embeddings']

    unlabeled_embeddings = unlabeled_embeddings.to(device)
    labeled_embeddings = labeled_embeddings.to(device)
    m = unlabeled_embeddings.shape[0]

    if labeled_embeddings.shape[0] == 0:
        min_dist = torch.tile(float('inf'),m)
    else:
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings,p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0]
    
    idxs = []

    for i in range(n):
        idx = torch.argmax(min_dist)
        idxs.append(idx.item())
        dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
        min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])
            
    return idxs

def badge(unlabeled_grad_embedding, n, device, **kwags):
    unlabeled_grad_embedding = unlabeled_grad_embedding.cpu().numpy()
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in unlabeled_grad_embedding])
    mu = [unlabeled_grad_embedding[ind]]
    indsAll = [ind]
    centInds = [0.] * len(unlabeled_grad_embedding)
    cent = 0
    while len(mu) < n:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(unlabeled_grad_embedding).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(unlabeled_grad_embedding).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(unlabeled_grad_embedding)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(unlabeled_grad_embedding[ind])
        indsAll.append(ind)
        cent += 1

    return indsAll

class Uncertainty(object):
    @staticmethod
    def margin(out):
        out = nn.Softmax(-1)(out)
        top2 = torch.topk(out, 2).values
        return 1 - (top2[:, 0] - top2[:, 1])

    @staticmethod
    def entropy(out):
        out = nn.Softmax(-1)(out)
        epsilon = 1e-5
        entropy = -out * torch.log(out + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    @staticmethod
    def confidence(out):
        out = nn.Softmax(-1)(out)
        top1 = torch.max(out, dim=1).values
        return 1 - top1

def margin(embeddings, n, device, **kwags):
    uncertainty = Uncertainty.margin(embeddings.to(device))
    indx = torch.argsort(uncertainty, descending=True)
    return indx[:n].cpu().tolist()

def bvsb(embeddings, n, device, **kwags):
    uncertainty = Uncertainty.margin(embeddings.to(device))
    indx = torch.argsort(uncertainty, descending=True)
    return indx[:n].cpu().tolist()

def entropy(embeddings, n, device, **kwags):
    uncertainty = Uncertainty.entropy(embeddings.to(device))
    indx = torch.argsort(uncertainty, descending=True)
    return indx[:n].cpu().tolist()

def confidence(embeddings, n, device, **kwags):
    uncertainty = Uncertainty.confidence(embeddings.to(device))
    indx = torch.argsort(uncertainty, descending=True)
    return indx[:n].cpu().tolist()

def erm(embeddings, n, device, **kwags):
    indx = np.arange(embeddings.shape[0])
    np.random.shuffle(indx)
    return indx[:n].tolist()


def assign_label(all_fea, initc, discartc, class_num, domian_num=1):
    center_num = class_num * domian_num
    centers = np.arange(center_num)

    livec_index = np.setdiff1d(centers, discartc)
    livec = initc[livec_index]
    live_cl = (centers % class_num)[livec_index]

    dd = cdist(all_fea, livec, 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = live_cl[pred_label]

    return pred_label, dd, live_cl.shape[0]


def merge_rank(rank1, rank2, gamma):
    rank = (gamma * rank1 + (1 - gamma) * rank2)
    return rank


def merge_rank_multi(ranks, gammas):
    rank_all = 0
    for rank, gamma in zip(ranks, gammas):
        rank_all += (gamma * rank)
    return rank_all


def ours3(unlabeled_embedding, n, device, **kwags):
    domains = kwags['domains']
    d_num = kwags['domains_num']
    c_num = kwags['class_num']
    initcs = kwags['initcs']
    probs = kwags['probs']
    discartcs = kwags['discartcs']
    probs_d = kwags['probs_d']
    uncertainty_method = kwags['uncertainty']
    domainess_method = kwags['domainess']
    domainess_flip = kwags['domainess_flip']
    gamma = kwags['gamma']

    domains = domains.cpu().numpy()
    unlabeled_embedding = unlabeled_embedding.cpu()
    if unlabeled_embedding.size(1) != initcs.shape[1]:
        unlabeled_embedding = torch.cat((unlabeled_embedding, torch.ones(unlabeled_embedding.size(0), 1)), 1)
    unlabeled_embedding = unlabeled_embedding.numpy()
    sample_num = unlabeled_embedding.shape[0]

    uncertainty_idxs = uncertainty_method(probs, sample_num, device)
    uncertainty_rank = np.zeros(sample_num)
    uncertainty_rank[uncertainty_idxs] = np.arange(sample_num)

    domainess_idxs = domainess_method(probs_d, sample_num, device)
    domainess_rank = np.zeros(sample_num)
    if domainess_flip:
        domainess_rank[domainess_idxs] =  np.arange(sample_num - 1, -1, -1)
    else:
        domainess_rank[domainess_idxs] = np.arange(sample_num)

    pred_label, dd, K = assign_label(unlabeled_embedding, initcs, discartcs, c_num, d_num)
    distance = dd.min(axis=1)
    diversity_idxs = np.argsort(distance)[::-1]
    diversity_rank = np.zeros(sample_num)
    diversity_rank[diversity_idxs] = np.arange(sample_num)

    rank = merge_rank_multi([domainess_rank, diversity_rank, uncertainty_rank], gamma)

    rank_indx = np.argsort(rank)
    
    return list(rank_indx[:n])