import numpy.random as npr

def change_lab(target, n_class):
    """
    Generate a random integer between 0 and n_class that is not equal to target.
    
    Parameters:
        target (int): The target value.
        n_class (int): The total number of classes.
    
    Returns:
        int: The randomly generated integer.
    """
    res = npr.randint(n_class-1)
    if res<target:
        return res
    else: 
        return res+1

def give_noise(dataset, p):
    """
    Generate a noisy version of the given dataset.

    Parameters:
    - dataset: The original dataset.
    - p: The probability of adding noise to each data point.

    Returns:
    - new_dataset: The noisy version of the dataset.
    """
    n = len(dataset)
    n_class = len(dataset.classes)
    new_dataset = []
    for i in range(n):
        if npr.rand()<p:
            new_dataset.append((dataset[i][0], 
                                change_lab(dataset[i][1], n_class)))
        else: 
            new_dataset.append((dataset[i][0], dataset[i][1]))
    return new_dataset
