"""
module which contains train_fasterrnn and train functions
"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch_snippets import Report
train_batch = __import__('train_batches').train_batch
validate_batch = __import__('train_batches').validate_batch
prepare_loaders = __import__('prepare_loaders').prepare_loaders
prepare_model = __import__('load_model').prepare_model


def train_fasterrnn(model:fasterrcnn_resnet50_fpn,
                    optimizer:torch.optim.Optimizer,
                    n_epochs:int, train_loader:torch.utils.data.DataLoader,
                    test_loader:torch.utils.data.DataLoader,
                    log:Report, keys:list, device:str):
    """
    function which performs the training of
    - faster region-based neural network model
    - model: FasterRCNN
    - optimizer: torch optimizer used during gradient descent
    - n_epochs: number of epochs during training
    - train_loader: DataLoader instance for training data
    - test_loader: DataLoader instance for training data
    - log: torch_snippet record to record training progress
    - keys: list of keys for FasterRCNN loss names
    - device: string specifying device used for training
    Returns: log object and test_dl to test the model
    """
    if not log:
        log = Report(n_epochs)
    if not keys:
        keys = ['loss_classifier',
                'loss_box_reg',
                'loss_objectness',
                'loss_rpn_box_reg']
    model.to(device)

    
    for epoch in range(n_epochs):
        N = len(train_loader)
        for t, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model,
                                       optimizer, device)
            # record the current train loss
            pos = epoch + (t + 1) / N
            log.record(pos=pos, trn_loss=loss.item(),
                       end='\r')
        if test_loader:
            N = len(test_loader)
            for v, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model,
                                              optimizer, device)
                # record the current validation loss
                pos = epoch + (v + 1) / N
                log.record(pos=pos, trn_loss=loss.item(),
                           end='\r')
    log.report_avgs(epoch + 1)
    return log

def train(num_classes: int=2, labels: dict={}, feature_extraction: bool=False,
          learing_rate: float=.005, momentum=.9, weight_decay:float =.0005,
          n_epochs: int=1, batch_size: int=4, root: str='./', device:str=None) -> tuple:
    """
    prepares loaders, model and optimizer and calls the training function
    num_classes: num of classes to classify (2 is a binary classification)
    labels: labels to predict
    feature_extraction: determines when wheighs will be frozen during training
    learning_rate: step (importance) of the gradient during back propagation
    momentum: momentum used in stochastic gradient descent
    weight_decay: rate of decay of weights
    n_epochs: epochs in which we train the model
    batch_size: batch_size
    root: path to root folder
    device: device used during training
    Returns: log object, test_dl, model
    """
    train_dl, val_dl, test_dl = prepare_loaders(root, labels, batch_size)
    model = prepare_model(num_classes, feature_extraction)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params,
                                lr=learing_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    return train_fasterrnn(model=model, optimizer=optimizer, n_epochs=n_epochs,
                           train_loader=train_dl, test_loader=val_dl,
                           log=None, keys=None, device=device), test_dl, model
