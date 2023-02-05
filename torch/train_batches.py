import torch
import torchvision
"""
module which contains train_batch validate_batch and unbatch functions
"""

def unbatch(batch: torch.utils.data.DataLoader, device: str) -> tuple:
    """
    prepares the batch into an (X, y) tuple and prepares each element
    to be loaded in memory on the specified device
    """
    X, y = batch

    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y

def train_batch(batch: torch.utils.data.DataLoader,
                model:torchvision.models.detection,
                optimizer:torch.optim.Optimizer, device:str):
    """
    Uses back propagation to train the model
    it's just used in unfreezed layers
    - batch: tuple of (X, y) batch from DataLoader
    - model: torch model
    - optimizer: torch optimizer used for gradient descent
    - device: device in which the algorithm is trained
    Returns:
        - loss: sum of the batch losses
        - losses: dictionary containing individual losses
    """
    # model.train() tells to the model you're training it.
    # set the layers which their respectives behaivors
    # it's counterintuitively, but it doesn't train the model
    model.train()
    X, y = unbatch(batch, device)
    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
    looses = model(X, y)
    # adding all looses to call backward() once
    loss = sum([loss for loss in looses.values()])
    loss.backward() # computes backpropagation through gradient descent
    optimizer.step() # reevaluates the model through optimization
    return loss, looses

def validate_batch(batch: torch.utils.data.DataLoader,
                model:torchvision.models.detection,
                optimizer:torch.optim.Optimizer, device:str):
    """
    Validates the model
    it's just used in unfreezed layers
    - batch: tuple of (X, y) batch from DataLoader
    - model: torch model
    - optimizer: torch optimizer used for gradient descent
    - device: device in which the algorithm is trained
    Returns:
        - loss: sum of the batch losses
        - losses: dictionary containing individual losses
    """
    # model.train() tells to the model you're training it.
    # set the layers which their respectives behaivors
    # it's counterintuitively, but it doesn't train the model
    X, y = unbatch(batch, device)
    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
    looses = model(X, y)
    # adding all looses to call backward() once
    loss = sum([loss for loss in looses.values()])
    return loss, looses
