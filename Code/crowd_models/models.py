import torch
from torch.nn.functional import softmax
from setup_datasets import *
import  torch.nn as nn
from random import sample

def worker_pred(model, data, worker):
    """
    Load the trained weights of the model corresponding to the given worker and 
    compute its prediction.
    
    Parameters:
    - model: The model to load the weights into.
    - data: The input data to feed into the model.
    - worker: The identifier of the worker whose model weights to load.
    
    Returns:
    - Tensor: The softmax probabilities predicted by the model.
    """
    model.load_state_dict(torch.load(f'./expert_models/model-{worker}_weights.pth'))
    model.eval()
    return  softmax(model(data.to("cuda").float()), dim=1)

def get_img(img, dataset):
    """
    Find the index of the given image in the dataset.

    Parameters:
        img (torch.Tensor): The image to search for.
        dataset (list): The dataset containing images.

    Returns:
        int: The index of the image in the dataset.
    """
    i = 0
    while not(torch.equal(img, dataset[i][0])):
        i=i+1
    return i 

def diff(liste1, liste2):
    """
    Calculates the difference between two lists.

    Args:
        liste1 (list): The first list.
        liste2 (list): The second list.

    Returns:
        list: A list containing the elements that are in liste1 but not in liste2.
    """
    res = []
    for i in liste1:
        if i not in liste2:
            res.append(i)
    return res

def get_workers(img, dataset, json, tot_worker):
    """
    Get the workers of a given image in a dataset.

    Parameters:
        img (str): The image to retrieve workers for.
        dataset (str): The dataset containing the image.
        json (dict): The JSON data containing worker information.
        tot_worker (int): All the workers you want to use.

    Returns:
        list: A list of workers who have worked on the given image.

    """
    task = get_img(img, dataset)
    worker = list(json[f"{task}"].keys())
    return diff(tot_worker, worker)


class Weight(nn.Module):
    def __init__(self, n_worker):
        super(Weight,self).__init__()
        w = torch.full(size=tuple([n_worker]), fill_value=(1/n_worker))
        self.weights = nn.Parameter(w)

    def forward(self, x, select_w):
        """
        Calculate the forward pass of the weight network.

        Parameters:
            x (torch.Tensor): The input tensor.
            select_w (torch.Tensor): The indicative tensor for selective weights.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        final = x.t().matmul(self.weights)
        if self.training:
            final = (1/select_w.t().matmul(self.weights))*final
        return final

class CIFAR10H(nn.Module):
    def __init__(self):
        super(CIFAR10H,self).__init__()
        self.n_worker = 2571
        self.n_classe = 10
        self.weight = Weight(self.n_worker)
        self.traintask = load_crowd(cifar10h_root, type='train')
        self.expert = worker_json(self.traintask).keys()

    def get_workers(self, num):
        """
        Initializes the `workers` attribute with a random sample of `num` elements from the `expert` attribute.

        Parameters:
            num (int): The number of workers to be selected.

        Returns:
            None
        """
        self.workers = sample(list(self.expert), num)

    def forward(self, x, dataset, model):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input data of shape (batch_size, input_size).
            dataset (Dataset): The dataset object containing the training data.
            model (torch.nn.Module): The model used for prediction of the workers.

        Returns:
            torch.Tensor: The predicted output of shape (batch_size, n_classes).
        """
        self.weight.training = self.training
        batch_size = len(x)
        pred = torch.zeros((batch_size, self.n_classe)).to("cuda")
        for i in range(batch_size):
            work_t = get_workers(x[i], dataset, self.traintask, self.workers)
            p = torch.zeros((self.n_worker, self.n_classe))
            sel_w = torch.zeros((self.n_worker, 1))
            for k in work_t:
                index = self.workers.index(k)
                p[index] = worker_pred(model, x, int(k))[i].detach()
                sel_w[index] = 1
            pred[i] = self.weight(p, sel_w)
        return pred
