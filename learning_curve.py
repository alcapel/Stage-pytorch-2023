#%%
import torch
from setup_cifar10h import worker_json, dl_link
import json
from peerannot.helpers.networks import networks
from .pytorch_tutorial.programme.loop import train, val

root = "../../peerannot/datasets/cifar10H/"
#%% Construction dataset pour l'entrainement du worker 0
train_file = open(root+"answers.json","r")
tContent = train_file.read()
obj_train = json.loads(tContent)   

t = worker_json(obj_train, get_nb=False)
tw0 = t["0"]
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
dataset_tw0 = []

for i in list(tw0.keys()): 
    dataset_tw0.append(dl_link(tw0[i],int(i),classe))

#%% Construction dataset pour la validation du worker 0
val_file = open(root+"answers_valid.json","r")
vContent = val_file.read()
obj_val = json.loads(vContent)   

v = worker_json(obj_val, get_nb=False)
vw0 = v["0"]
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
dataset_vw0 = []

for i in list(vw0.keys()): 
    dataset_vw0.append(dl_link(vw0[i],int(i),classe, type='val'))


#%%
from torch.utils.data import DataLoader
trainsetw0 = DataLoader(dataset_tw0,  batch_size=8, shuffle=True, num_workers=2)
valsetw0 = DataLoader(dataset_vw0, batch_size=8, shuffle=True, num_workers=2)

# %%
Net = networks('resnet18', n_classes=10, pretrained=True)

# %% Gel des paramètres sauf sur la dernière couche 

# for param in Net.parameters():
#     param.requires_grad = False

# for param in Net.fc.parameters():
#     param.requires_grad = True

#%% 
# Freeze all layers except the last one
for name, param in Net.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False


# %%
import torch.nn as nn
import torch.optim as optim
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
# %%
