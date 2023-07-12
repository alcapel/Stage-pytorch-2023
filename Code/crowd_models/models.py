import torch
from torch.nn.functional import softmax


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