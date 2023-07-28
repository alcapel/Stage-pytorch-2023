#%%
from setup_datasets import dl_link, label_dist
import torch
from torch.nn.functional import softmax
from peerannot.helpers.networks import networks
import torch.optim as optim
import torch.nn as nn
from models import CIFAR10H
import gc
import numpy as np

# %%
get_weights = CIFAR10H()
optimizer = optim.SGD(get_weights.parameters(), lr=0.01, momentum=0.9)
loss = nn.CrossEntropyLoss()

#%%
lab = label_dist(get_weights.traintask, 10)
classe = ["plane","cat","dog","car","deer", "horse","ship",
          "truck","frog", "bird"]
n = len(get_weights.traintask)
dataset = []
for i in range(n):
    dataset.append(dl_link(lab[i], i, classe))

#%%
from torch.utils.data import DataLoader
batch_size = 32
trainset = DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=2)

# %%
get_weights.get_workers(100)
err = np.zeros(len(trainset))
for batch, (X,y) in enumerate(trainset):
    optimizer.zero_grad()
    model = networks('resnet18', n_classes=10, pretrained=False).to("cuda")
    logits = get_weights(X, dataset, model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    lossRes = loss(logits, y.to("cuda"))
    err[batch] = lossRes.item()
    lossRes.backward()
    optimizer.step()


# %%
