#%%
import torch
from setup_datasets import *
from peerannot.helpers.networks import networks
from ..pytorch_tutorial.programme.loop import train, val
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#%% Construction dataset pour l'entrainement du worker 0
obj_train = load_crowd(cifar10h, type='train')
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]

t = worker_json(obj_train, get_nb=False)
tw0 = t["0"]
dataset_tw0 = []

for i in list(tw0.keys()): 
    dataset_tw0.append(dl_link(tw0[i],int(i),classe))

#%% Construction dataset pour la validation du worker 0
obj_val = load_crowd(cifar10h, type='valid')

v = worker_json(obj_val, get_nb=False)
vw0 = v["0"]
dataset_vw0 = []

for i in list(vw0.keys()): 
    dataset_vw0.append(dl_link(vw0[i],int(i),classe, type='val'))


#%%
trainsetw0 = DataLoader(dataset_tw0,  batch_size=8, shuffle=True, num_workers=2)
valsetw0 = DataLoader(dataset_vw0, batch_size=8, shuffle=True, num_workers=2)

# %% Gel des paramètres sauf sur la dernière couche 
# Net = networks('resnet18', n_classes=10, pretrained=True)

# for param in Net.parameters():
#     param.requires_grad = False

# for param in Net.fc.parameters():
#     param.requires_grad = True

#%% 
Net = networks('resnet18', n_classes=10, pretrained=True)

# Freeze all layers except the last one
for name, param in Net.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False

# %%
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)
epochs = 10

# Apprentissage et tests
l_curve = torch.zeros(epochs)
v_curve = torch.zeros(epochs)

for i in range(epochs):
    print(f"Boucle n°{i+1} ------------------------------\n")
    l_curve[i] = train(trainsetw0, Net, optimizer, loss)
    v_curve[i] = val(valsetw0, Net, loss)
print("C'est terminé !")

# %%
import matplotlib.pyplot as plt

plt.plot(l_curve)
plt.plot(v_curve)
