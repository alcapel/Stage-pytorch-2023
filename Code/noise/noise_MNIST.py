#%%
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ..pytorch_tutorial.programme.loop import train, test
from corruption import give_noise
import numpy as np

# %% Réseau de neurone simple.
class test_model(nn.Module):
    '''
    Réseau de neurone emprunter sur Pytorch.org.
    '''
    def __init__(self):
        super(test_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% Importation des datasets
trainset = MNIST(root='./data', train=True, download=True,
                  transform=transforms.ToTensor())
testset = MNIST(root='./data', train=False, download=True,
                  transform=transforms.ToTensor())
testMNIST = DataLoader(testset, batch_size=32,
                       shuffle=False, num_workers=2)

# %%
prob = np.arange(0, 1, 0.01)
epochs = 20
loss = nn.CrossEntropyLoss()
err = np.zeros(len(prob))
#%%
j = 0
for p in prob:
    print(f"On fait p = {p}.", end="\r")
    MNISTload = DataLoader(give_noise(trainset, p),
                           batch_size=32,
                           shuffle=True, 
                           num_workers=2)
    model = test_model().to("cuda")
    optimizer = optim.SGD(model.parameters(), 
                          lr=0.001, 
                          momentum=0.9)
    for i in range(epochs):
        train(MNISTload,
              model, 
              optimizer, 
              loss, 
              ongoing=False)
    err[j] = test(testMNIST, 
                  model)
    j+=1
    
with open("./err_MNIST.npy",'wb') as f:
    np.save(f, err)

# %% Chargement des erreurs
with open("./error_MNIST.npy",'rb') as f:
    err = np.load(f)

#%% Affichage des erreurs
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=prob, y=1-err))
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

#%% Téléchargement du graphe
fig.write_html(".\graph_MNIST.html")


