#%%
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ..pytorch_tutorial.programme.loop import train, test
from peerannot.helpers.networks import networks
from corruption import give_noise
import numpy as np

#%% Importation des datasets
trainset = CIFAR10(root='./data', train=True, download=True,
                  transform=transforms.ToTensor())
testset = CIFAR10(root='./data', train=False, download=True,
                  transform=transforms.ToTensor())
testCIFAR10 = DataLoader(testset, batch_size=64,
                         shuffle=False, num_workers=2)

# %%

trainloss = np.zeros(100)
valloss = np.zeros(100)
prob = np.arange(0, 1, 0.01)
epochs = 30
loss = nn.CrossEntropyLoss()


#%%
# CIFAR10load = DataLoader(trainset,
#                         batch_size=32,
#                         shuffle=True, 
#                         num_workers=2)
# model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")
# optimizer = optim.SGD(model.parameters(), 
#                         lr=0.001, 
#                         momentum=0.9)


# for i in range(epochs):
#     print(f"Ma progression : {i}/100")
#     trainloss[i] = train(CIFAR10load,
#                          model, 
#                          optimizer, 
#                          loss, 
#                          ongoing=False)
#     valloss[i] = val(testCIFAR10, model, loss)

# #%%
# import matplotlib.pyplot as plt
# plt.plot(trainloss)
# plt.plot(valloss)

#%%

err = np.zeros(len(prob))
j = 0
for p in prob:
    print(f"On fait p = {p}.", end="\r")
    CIFAR10load = DataLoader(give_noise(trainset, p),
                           batch_size=64,
                           shuffle=True, 
                           num_workers=2)
    model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")
    optimizer = optim.SGD(model.parameters(), 
                          lr=0.001, 
                          momentum=0.9)
    for i in range(epochs):
        train(CIFAR10load,
              model, 
              optimizer, 
              loss, 
              ongoing=False)
    err[j] = test(testCIFAR10, 
                  model)
    j+=1
    
with open("./err_cifar10.npy",'wb') as f:
    np.save(f, err)


# %%
import numpy as np

with open("./error_cifar10.npy",'rb') as f:
    err = np.load(f)

#%%

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=prob, y=1-err-0.3))
fig.update_layout(
    xaxis_title="Probabilité que le label soit incorrect",
    yaxis_title="Taux d'erreur de classification",
    title="L'élève dépasse le maître ?",
    title_x=0.5,
    paper_bgcolor='rgb(230,230,230)',

    title_font=dict(color='black')
)
fig.update_layout(
    xaxis=dict(
        range=[0, 1],  
        autorange=False 
    ),
    yaxis=dict(
        range=[0, 1],  
        autorange=False 
    )
)
fig.update_layout(
    xaxis=dict(
        dtick=0.1  
    ),
    yaxis=dict(
        dtick=0.1 
    )
)

fig.layout.xaxis.color = 'black'
fig.layout.yaxis.color = 'black'
fig.update_traces(line=dict(color="darkblue"))
fig.show()

#%%
fig.write_html("./graph_cifar10.html")
