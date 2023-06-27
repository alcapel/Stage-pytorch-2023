#%% 
import json
fileObject = open("../peerannot/datasets/cifar10H/answers.json","r")
root = "../peerannot/datasets/cifar10H/"
# %%
jsContent = fileObject.read()
obj_python = json.loads(jsContent)   
# %%
print(obj_python)
# %%
print(obj_python['9500'])
# %%
len(obj_python)
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

# %%
a, b = worker_json(obj_python, get_nb=True)
# %% Vérification si on a bien tout
check = []
c = 0
for k, dict in a.items() :
    c+= len(dict)
    for im, lab in dict.items():
        if im not in check:
            check.append(im)
print((b==c)&(len(check)==len(obj_python)))   
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
import torch
def get_prob(list, n_class):
    '''
    Donne le tenseur de la distribution des labels de list.
    '''
    prob = torch.zeros(n_class)
    for i in range(n_class):
        prob[i] = count(list, i)
    return prob/len(list)
# %%
test = list(obj_python["0"].values())
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


# %%

p_tensor = label_dist(obj_python, 10)
p_dict =  label_dist(obj_python, 10, which='dict')
# %% Lecture image avec torchvision
from torchvision.io import read_image
tsr_img = read_image(root+"train/cat/cat-0.png")
# %%
import os 

os.path.exists(root+f"train/cat/cat-0.png")

#%%
def dl_link(lab, num, classe):
    '''
    Dans la configuration d'un dataset peerannot, retourne un tuple liant la donnée avec
    le label (ou la distribution).
    '''
    for cl in classe:
        if os.path.exists(root+f"./train/{cl}/{cl}-{num}.png"):
            return (read_image(root+f"./train/{cl}/{cl}-{num}.png"), lab)
            
#%% Construction d'une liste de tuple de dimension 2, avec l'image et la distribution donnée
# par les workers


classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
dataset = []
for i in range(len(p_tensor)):
    dataset.append(dl_link(p_tensor[i], i, classe))


#%% Construction dataset pour l'entrainement du worker 0

test = a["0"]
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
dataset_w0 = []

for i in list(test.keys()): 
    dataset_w0.append(dl_link(test[i],int(i),classe))


#%%
from torch.utils.data import DataLoader
trainset = DataLoader(dataset,  batch_size=2, shuffle=True, num_workers=2)
trainsetw0 = DataLoader(dataset_w0,  batch_size=4, shuffle=True, num_workers=2)


# %% Maintenant nous allons essayer de faire un dataset pour chaque worker

dataset_worker = []
n = len(a)

for i in range(n):
    inter = []
    for task, lab in a[f"{i}"].items():
        inter.append(dl_link(lab, int(task), classe))
    dataset_worker.append(inter)

# %%
from peerannot.helpers.networks import *
# %%
Net = networks('resnet18', n_classes=10, pretrained=False)
# %%
def train(trainloader, net, optimizer, loss):
    ''' 
    Permet d'entrainer à partir d'un DataLoader 
    (trainloader) un réseau (net) en utilisant 
    la fonction de perte loss.
    
    '''
    net.train()
    for batch, (X,label) in enumerate(trainloader):
        optimizer.zero_grad()
        logits = net(X)
        lossRes = loss(logits, label)
        lossRes.backward()
        optimizer.step()
        if batch%100==0 :
            print(f'Perte = {lossRes.item()} ',
                  f'Avancée : [{(batch+1)*len(X)}/{len(trainloader.dataset)}]')

# %%
import torch.optim as optim
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
epochs = 2

# Apprentissage et tests
for i in range(epochs):
    print(f"Boucle n°{i+1} ------------------------------\n")
    train(trainsetw0, Net, optimizer, loss)

print("C'est terminé !")
# %%
