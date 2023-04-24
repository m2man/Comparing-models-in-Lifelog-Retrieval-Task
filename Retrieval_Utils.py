import numpy as np
import torch
from collections import OrderedDict

def evaluate_recall(sims, mode='i2t', answer_each=5, groups_dict_i2t=None, groups_dict_t2i=None):
    if mode == 'i2t':
        recall, _ = i2t(sims, return_ranks=False, answer_each=answer_each, groups_dict=groups_dict_i2t)
        r1i, r5i, r10i, _, _ = recall
        r1t, r5t, r10t = None, None, None
    if mode == 't2i':
        recall, _ = t2i(sims, return_ranks=False, answer_each=answer_each, groups_dict=groups_dict_t2i)
        r1t, r5t, r10t, _, _ = recall
        r1i, r5i, r10i = None, None, None
    if mode == 'both':
        recall_i2t, _ = i2t(sims, return_ranks=False, answer_each=answer_each, groups_dict=groups_dict_i2t)
        recall_t2i, _ = t2i(sims, return_ranks=False, answer_each=answer_each, groups_dict=groups_dict_t2i)
        r1i, r5i, r10i, _, _ = recall_i2t
        r1t, r5t, r10t, _, _ = recall_t2i
    return r1i, r5i, r10i, r1t, r5t, r10t

def i2t(sims, return_ranks=False, answer_each=5, groups_dict=None):
    # groups_dict with keys is image_id, value is matched_text_id (or vice versa)
    # sims (n_imgs, n_caps)
    n_imgs, n_caps = sims.shape
    ranks = np.zeros(n_imgs)
    top1 = np.zeros(n_imgs)
    results = []
    for index in range(n_imgs):
        result = dict()
        result['id'] = index
        inds = np.argsort(sims[index])[::-1]
        result['top5'] = list(inds[:5])
        result['top1'] = inds[0]
        result['top10'] = list(inds[:10])
        result['ranks'] = []
        # Score
        rank = 1e20
        if groups_dict is not None:
            label = groups_dict[index] # list of index matched index of the current image
            for i in label:
                tmp = np.where(inds == i)[0][0]
                result['ranks'].append((i, tmp))
                if tmp < rank:
                    rank = tmp
        else:
            for i in range(answer_each * index, answer_each * (index + 1), 1):
                tmp = np.where(inds == i)[0][0]
                result['ranks'].append((i, tmp))
                if tmp < rank:
                    rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if rank<1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if rank<5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0

        results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results

def t2i(sims, return_ranks=False, answer_each=5, groups_dict=None):
    if groups_dict is not None:
        return i2t(sims.T, return_ranks, answer_each, groups_dict)
    else:
        # sims (n_imgs, n_caps)
        n_imgs, n_caps = sims.shape
        ranks = np.zeros(answer_each*n_imgs)
        top1 = np.zeros(answer_each*n_imgs)
        # --> (5N(caption), N(image))
        sims = sims.T # ncap, nimg
        results = []
        for index in range(n_imgs):
            for i in range(answer_each):
                result = dict()
                result['id'] = answer_each*index+i
                inds = np.argsort(sims[answer_each * index + i])[::-1]
                result['top5'] = list(inds[:5])
                result['top10'] = list(inds[:10])
                result['top1'] = inds[0]
                ranks[answer_each * index + i] = np.where(inds == index)[0][0]
                top1[answer_each * index + i] = inds[0]

                if ranks[answer_each*index+i]<1:
                    result['is_top1'] = 1
                else:
                    result['is_top1'] = 0

                if ranks[answer_each*index+i] <5:
                    result['is_top5'] =1
                else:
                    result['is_top5'] = 0
                result['ranks'] = [(index, ranks[answer_each*index+i])]
                results.append(result)

        # Compute metrics
        r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        if return_ranks:
            return (r1, r5, r10, medr, meanr), (ranks, top1),results
        else:
            return (r1, r5, r10, medr, meanr), results