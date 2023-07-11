#%%
import torch
import torch.nn as nn
import torch.optim as optim
from setup_datasets import worker_json, dl_link, cifar10h
import json
from peerannot.helpers.networks import networks
from ..pytorch_tutorial.programme.loop import train
from torch.utils.data import DataLoader

#%% Chargement des données et fonctions 
train_file = open(cifar10h+"answers.json","r")
tContent = train_file.read()
obj_train = json.loads(tContent)   
t = worker_json(obj_train, get_nb=False)
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
loss = nn.CrossEntropyLoss()
epochs = 100

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

def gel(net):
    for param in net.parameters():
        param.requires_grad = False

    for param in net.fc.parameters():
        param.requires_grad = True

#%% Mise en place de tous les datasets

dataset_all = []
n = len(t)

for i in range(n):
    inter = []
    for task, lab in t[f"{i}"].items():
        inter.append(dl_link(lab, int(task), classe))
    dataset_all.append(inter)
print("C'est fini!")

#%% Entrainement des modèles

for i in range(n):
    print(i, end="\r")
    if len(dataset_all[i])%32==1:
        trainset = DataLoader(dataset_all[i],  batch_size=30, 
                              shuffle=True, num_workers=2)
    else:
        trainset = DataLoader(dataset_all[i],  batch_size=32, 
                              shuffle=True, num_workers=2)
    Net = networks('resnet18', n_classes=10, pretrained=True).to("cuda")
    gel(Net)
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)
    for j in range(epochs):
        train(trainset, Net, optimizer, loss, ongoing=False)
    torch.save(Net.state_dict(), f"/home/acapel/programme/programme/expert_models/model-{i}_weights.pth")

print("Tout le monde est entrainé !")
