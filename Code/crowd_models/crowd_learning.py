#%% 
import torch
from ..pytorch_tutorial.programme.loop import train, test
from setup_datasets import label_dist, dl_link, worker_json, cifar10h
import json
from peerannot.helpers.networks import networks
from torch.nn.functional import softmax
import torch.optim as optim
import torch.nn as nn


# %%
root = "../../peerannot/datasets/cifar10H/"
train_file = open(cifar10h+"answers.json","r")
tContent = train_file.read()
obj_train = json.loads(tContent) 
#%%
lab = label_dist(obj_train, 10)
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
n_classe = len(classe)
# %%
t = worker_json(obj_train, get_nb=False)
select_w = []
for i in range(2571):
    if len(t[f'{i}'])>195:
        select_w.append(i)
print("voila")

#%%
n = len(obj_train)
dataset = []
for i in range(n):
    dataset.append(dl_link(lab[i], i, classe))

# %%
from torch.utils.data import DataLoader
batch_size = 32
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
w = torch.ones(37, dtype=float, requires_grad=True)/37
param = nn.Parameter(w)


#%%

def get_img(img, dataset):
    i = 0
    while not(torch.equal(img, dataset[i][0])):
        i=i+1
    return i 

def where(k, tenseur):
    if k not in tenseur:
        print("Cet élément n'existe pas")
    else:
        arg = 0
        while (tenseur[arg]!=k)&(arg<len(tenseur)):
            arg+=1
        return arg

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
optimizer = optim.SGD([param], lr=0.01,momentum=0.9)
loss = nn.CrossEntropyLoss()

for batch, (X,y) in enumerate(trainset):
    optimizer.zero_grad()
    p = torch.zeros(batch_size, n_classe)
    for i in range(batch_size):
        im = get_img(X[i], dataset)
        sum_w = 0
        for k in obj_train[f"{im}"].keys():
            if int(k) in select_w:
                arg = where(int(k), select_w)
                liste_mod[arg].train()
                logits = liste_mod[arg](X[i].to("cuda").float())
                p[i] += w[arg]*softmax(logits)
                sum_w += w[arg]
        p[i] = p[i]/sum_w
    lossRes = loss(p,y)
    lossRes.backward()
    optimizer.step()

# %%
