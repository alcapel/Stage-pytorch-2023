import torch

def train(trainloader, net, optimizer, loss):
    ''' 
    Permet d'entrainer à partir d'un DataLoader 
    (trainloader) un réseau (net) en utilisant 
    la fonction de perte loss.
    
    '''
    net.train()
    loss_e = 0
    for batch, (X,label) in enumerate(trainloader):
        optimizer.zero_grad()
        logits = net(X.float())
        lossRes = loss(logits, label)
        loss_e += lossRes.item()
        lossRes.backward()
        optimizer.step()
        if batch%20==0 :
            print(f'Perte = {lossRes.item()} ',
                  f'Avancée : [{(batch+1)*len(X)}/{len(trainloader.dataset)}]')
    return loss_e/len(trainloader)

def val(valloader, net, loss):
    net.eval()
    loss_e = 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(valloader):
            logits = net(X.float())
            loss_e += loss(logits, y).item()
    return loss_e/len(valloader)




def test(testloader, net):
    '''
    Permet de tester le réseau net avec Dataloader test.
    Affiche la précision du modèle en sortie.
    
    '''
    correct = 0
    tot = len(testloader.dataset)
    net.eval()
    with torch.no_grad():
        for (X,y) in testloader:
            pred = net(X).logits
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"Précision du modèle : {100*(correct/tot)}%\n")
