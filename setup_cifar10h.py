import torch
import os 
from torchvision.io import read_image
root = "../../peerannot/datasets/cifar10H/"
# %% Pour avoir un dictionnaire avec
# d'abord les workers, et ensuite les tasks

def worker_json(answer, get_nb=False):
    w_answer = {}
    b = 0
    for task, dict in answer.items():
        b+=len(dict)
        for worker, classes in dict.items():
            if worker in w_answer:
                w_answer[worker][task]=classes 
            else:
                w_answer[worker] = {}
                w_answer[worker][task]=classes
    if get_nb:
        return w_answer, b
    return w_answer

# %% Vérification si on a bien tout
# a, b = worker_json(obj_python, get_nb=True)
# check = []
# c = 0
# for k, dict in a.items() :
#     c+= len(dict)
#     for im, lab in dict.items():
#         if im not in check:
#             check.append(im)
# print((b==c)&(len(check)==len(obj_python)))   
# %% 
def count(list, target):
    '''
    Pour compter le nombre de target dans list.
    '''
    n = len(list)
    res = 0
    for i in range(n):
        if list[i]==target:
            res+=1
    return res

# %%
def get_prob(list, n_class):
    '''
    Donne le tenseur de la distribution des labels de list.
    '''
    prob = torch.zeros(n_class)
    for i in range(n_class):
        prob[i] = count(list, i)
    return prob/len(list)
# %%
def label_dist(dict, n_class, which='tensor'):
    '''
    Crée un tenseur (ou un dictionnaire) qui pour chaque image ressort 
    la distribution de probabilité des votes des workers
    '''
    n = len(dict)
    keys = list(dict.keys())
    if which=='tensor':
        res = torch.zeros(n,n_class)
        for i in range(n):
            res[i] = get_prob(list(dict[keys[i]].values()), n_class)
        return res
    elif which=='dict':
        res = {}
        for i in range(n):
            res[keys[i]]=get_prob(list(dict[keys[i]].values()), n_class)
        return res

#%%
def dl_link(lab, num, classe, type='train'):
    '''
    Dans la configuration d'un dataset peerannot, retourne un tuple liant la donnée avec
    le label (ou la distribution).
    '''
    for cl in classe:
        if os.path.exists(root+type+f"/{cl}/{cl}-{num}.png"):
            return (read_image(root+type+f"/{cl}/{cl}-{num}.png"), lab)
            
#%% Construction d'une liste de tuple de dimension 2, avec l'image et la distribution donnée
# # par les workers

# classe = ["plane","cat","dog","car","deer", "horse","ship",
#           "truck","frog", "bird"]
# dataset = []
# for i in range(len(p_tensor)):
#     dataset.append(dl_link(p_tensor[i], i, classe))

# %% Maintenant nous allons essayer de faire un dataset pour chaque worker
# dataset_worker = []
# n = len(a)

# for i in range(n):
#     inter = []
#     for task, lab in a[f"{i}"].items():
#         inter.append(dl_link(lab, int(task), classe))
#     dataset_worker.append(inter)

