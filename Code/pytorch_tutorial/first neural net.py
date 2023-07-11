import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#print(f"Using {device} device")


## Création de la classe
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
#print(model)


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Décortiquons la construction de la classe étape par étape

input_image = torch.rand(3,28,28)

## nn.Flatten
# Transforme l'image (28x28) en un array (28x28=784 valeurs ici)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

## nn.Linear
# Applique une transformation linéaire (pour quoi faire ?)

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

## Relu
# Introduit de la non linéarité au modèle (après linear transformation)

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

## nn.Sequential 
# Permet de faire passer les données à travers tous les 
# modules précédents (dans l'ordre écrit) en une commande

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print(f"After Sequential: {logits}")

## nn.Softmax
# pour accéder aux probabilités d'appartenance à chaque classe

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"After Softmax: {sum(pred_probab[0])}")


## On peut avoir accès aux paramètres du réseaux de neurones (qui s'ajuste pendant l'apprentissage)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")