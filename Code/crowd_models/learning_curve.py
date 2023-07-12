#%%
import torch
from setup_datasets import *
import json
from peerannot.helpers.networks import networks
from ..pytorch_tutorial.programme.loop import train, val
from random import randint, seed

seed(10)
worker = str(randint(0, 2570))

#%% Construction dataset pour l'entrainement du worker 0
obj_train = load_crowd(cifar10h, type='train')  

t = worker_json(obj_train, get_nb=False)
tw0 = t[worker]
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
dataset_tw0 = []

for i in list(tw0.keys()): 
    dataset_tw0.append(dl_link(tw0[i],int(i),classe))

#%% Construction dataset pour la validation du worker 0
obj_val = load_crowd(cifar10h, type='valid')  

v = worker_json(obj_val, get_nb=False)
vw0 = v[worker]
dataset_vw0 = []

for i in list(vw0.keys()): 
    dataset_vw0.append(dl_link(vw0[i],int(i),classe, type='val'))


#%%
from torch.utils.data import DataLoader
trainsetw0 = DataLoader(dataset_tw0,  batch_size=30,
                        shuffle=True, num_workers=2)
valsetw0 = DataLoader(dataset_vw0, batch_size=30, 
                      shuffle=True, num_workers=2)

# %%
Net = networks('resnet18', n_classes=10, pretrained=True).to("cuda")
Net2 = networks('resnet18', n_classes=10, pretrained=True).to("cuda")

# %% Gel des paramètres sauf fc 
for param in Net.parameters():
    param.requires_grad = False

for param in Net.fc.parameters():
    param.requires_grad = True

#%% 
# Freeze all layers except the last one
for name, param in Net2.named_parameters():
    if "layer4" not in name:
        param.requires_grad = False

# %%
import torch.nn as nn
import torch.optim as optim
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(Net2.parameters(), lr=0.001, momentum=0.9)
epochs = 500

# Apprentissage et tests
l_curve = torch.zeros(epochs)
v_curve = torch.zeros(epochs)
l_curve2 = torch.zeros(epochs)
v_curve2 = torch.zeros(epochs)

for i in range(epochs):
    print(f"Boucle n°{i+1} ------------------------------\n")
    l_curve[i] = train(trainsetw0, Net, optimizer, loss, ongoing=False)
    v_curve[i] = val(valsetw0, Net, loss)
    l_curve2[i] = train(trainsetw0, Net2, optimizer2, loss, ongoing=False)
    v_curve2[i] = val(valsetw0, Net2, loss)
print("C'est terminé !")


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Créer la figure avec deux sous-figures côte à côte
fig = make_subplots(rows=1, cols=2, subplot_titles=("Pour Net", "Pour Net2"))

# Ajouter la première courbe à la première sous-figure
fig.add_trace(go.Scatter(y=l_curve, name='Courbe apprentissage', 
                         line=dict(color='blue')), 
              row=1, col=1)
fig.add_trace(go.Scatter(y=v_curve, name='Courbe validation',
                         line=dict(color='orange')), 
              row=1, col=1)

# Ajouter la deuxième courbe à la deuxième sous-figure
fig.add_trace(go.Scatter(y=l_curve2, name='Courbe apprentissage',
                         line=dict(color='blue')), 
              row=1, col=2)
fig.add_trace(go.Scatter(y=v_curve2, name='Courbe validation', 
                         line=dict(color='orange')),
              row=1, col=2)

# Mettre des titres aux sous-figures
fig.update_layout(
    title_text="Comparaison courbe d'apprentissage ",
    title_font_size=24,
    title_x=0.5,  # Centrer le titre
    showlegend=False  # Cacher la légende
)
fig.update_yaxes(matches='y')

# Afficher la figure
fig.show()
fig.write_html(".\learning_curve.html")
# ici on voit que l'on peut prendre juste 100 epochs pour avoir
# un résultat convenable. De plus, la courbe de validation de Net est équivalente
# à celle de Net2 qui demande plus de paramètres à optimiser. Autant utiliser 