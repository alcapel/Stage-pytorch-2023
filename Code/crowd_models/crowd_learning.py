#%% 
import torch
from setup_datasets import *
from torch.nn.functional import softmax
from peerannot.helpers.networks import networks
import torch.optim as optim
import torch.nn as nn
from models import worker_pred
import gc

# %%
obj_train = load_crowd(cifar10h, 'train')
lab = label_dist(obj_train, 10)
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
n_classe = len(classe)

# %%
t = worker_json(obj_train, get_nb=False)
n_worker = len(t)


#%% Création dataset avec les distribution
n = len(obj_train)
dataset = []
for i in range(n):
    dataset.append(dl_link(lab[i], i, classe))

# %%
from torch.utils.data import DataLoader
batch_size = 16
trainset = DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=2)

# %%
# liste_mod = []
# for i in range(2571):
#     model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")
#     model.load_state_dict(torch.load(f'./expert_models/model-{i}_weights.pth'))
#     liste_mod.append(model)

# print("Est ce qu'il y a tous les modèles ?", len(liste_mod)==2571)
#%%
# On a au plus 63 personnes qui ont voté sur une image et 
# une personne a voté au moins sur 181 images.
w = torch.ones(n_worker, dtype=float, requires_grad=True)/n_worker
param = nn.Parameter(w)


#%%
def get_img(img, dataset):
    i = 0
    while not(torch.equal(img, dataset[i][0])):
        i=i+1
    return i 

#%%
# Datajouet 
liste = []

for i in range(1000):
    lab = torch.rand(10)
    lab /= lab.sum()
    x = torch.rand((10,37))
    liste.append((x,lab))

datatest = DataLoader(liste, batch_size=batch_size,
                      shuffle=True, num_workers=2)
#%%
import torch.optim as optim
optimizer = optim.SGD([param], lr=0.01,momentum=0.9)
loss = nn.CrossEntropyLoss()

for j in range(50):
    for batch, (X, y) in enumerate(datatest):
        p = torch.zeros((32,10))
        if batch<31:
            optimizer.zero_grad()
            for i in range(batch_size):
                p[i] = X[i].matmul(param.float())
            lossRes = loss(p,y)
            lossRes.backward()
            optimizer.step()

#%%
k=0
count = 0
for i in range(2571):
    for j in range(2571):
        if (i<j):
            a = list(set(t[f"{i}"].keys())&set(t[f"{j}"].keys()))
            if len(a)>k:
                if len(a)==len(t[f"{i}"].keys()):
                    count+=1
                else:   
                    k=len(a)
                    print(i, j, k)
#%%

k=0
liste = []
for i in range(2571):
    for j in range(2571):
        if (i<j)&(i not in liste):
            a = list(set(t[f"{i}"].keys())&set(t[f"{j}"].keys()))
            if len(a)==len(t[f"{i}"].keys()):
                count+=1
                liste.append(j)

# %%
import torch.optim as optim
w = torch.ones(n_worker, dtype=float, requires_grad=True)/n_worker
param = nn.Parameter(w, requires_grad=True)
optimizer = optim.SGD([param], lr=0.01,momentum=0.9)
loss = nn.CrossEntropyLoss()
model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")

for batch, (X, lab) in enumerate(trainset):
    optimizer.zero_grad()
    p = torch.zeros((batch_size, n_classe), requires_grad=False).to("cuda")
    for i in range(batch_size):
        task = get_img(X[i],dataset)   # on cherche à avoir accès à l'image pour pouvoir savoir qui a voté
        worker = list(obj_train[f"{task}"].keys())   # liste de qui vote
        tot = 0
        for k in worker:
            p[i] = p[i] + worker_pred(model, X, int(k))[i]*param[int(k)]    #prédiction du worker k sur l'item i
            tot = tot + param[int(k)]
        p[i] = p[i]/tot
    lossRes = loss(p, lab.to("cuda"))
    lossRes.backward()
    optimizer.step()
    # l'article ne précise pas si il faut softmax les poids 
    # pendant la boucle ou à la fin.

# %%
import torch.optim as optim
w = torch.ones(n_worker, dtype=float, requires_grad=True)/n_worker
param = nn.Parameter(w, requires_grad=True)
optimizer = optim.SGD([param], lr=0.01,momentum=0.9)
loss = nn.CrossEntropyLoss()

for batch, (X, lab) in enumerate(trainset):
    optimizer.zero_grad()
    pred = torch.zeros((batch_size, n_classe), requires_grad=False).to("cuda")
    for i in range(batch_size):
        model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")

        task = get_img(X[i],dataset)   # on cherche à avoir accès à l'image pour pouvoir savoir qui a voté
        worker = list(obj_train[f"{task}"].keys())   # liste de qui vote
        tot = 0
        p = torch.zeros((n_worker, n_classe), requires_grad=False)
        sel_w = torch.zeros((n_worker,1))
        for k in worker:
            p[int(k)] = worker_pred(model, X, int(k))[i] #prédiction du worker k sur l'item i
            sel_w[int(k)] = 1
        w = param.clone()
        pred[i] = p.t().matmul(w.float())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with torch.autograd.set_detect_anomaly(True):
        lossRes = loss(pred, lab.to("cuda"))
    lossRes.backward()
    optimizer.step()
    
    # l'article ne précise pas si il faut softmax les poids 
    # pendant la boucle ou à la fin.
#%%
import torch.optim as optim

w = torch.ones(n_worker, dtype=torch.float, requires_grad=True) / n_worker
param = nn.Parameter(w)
optimizer = optim.SGD([param], lr=0.01, momentum=0.9)
loss = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)

for batch, (X, lab) in enumerate(trainset):
    optimizer.zero_grad()
    pred = torch.zeros((batch_size, n_classe)).to("cuda")
    model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")
    for i in range(batch_size):
        task = get_img(X[i], dataset)
        worker = list(obj_train[f"{task}"].keys())
        tot = 0
        p = torch.zeros((n_worker, n_classe))
        sel_w = torch.zeros((n_worker, 1))
        for k in worker:
            p[int(k)] = worker_pred(model, X, int(k))[i].detach()
            sel_w[int(k)] = 1
        pred[i] = p.t().matmul(param)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    lossRes = loss(pred, lab.to("cuda"))
    lossRes.backward()
    optimizer.step()

# %%
