"""
module which contains predict_batch, predict and decode_prediction functions
"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
unbatch = __import__('train_batches').unbatch

@torch.no_grad
def predict_batch(batch: torch.utils.data.DataLoader,
                  model:fasterrcnn_resnet50_fpn, device:str) -> tuple:
    """
    test the performance of the model with the test dataset
    - batch: DataLoader instance containing the test batch
    - model: pretrained faster residual convolutional neural network
    - device: device in which we're working (GPU/CPU)
    Returns:
        - images: list of tensors of the images
        - predictions: a dict containing the predicted bounding boxes,
            labels and confidence scores
    """
    model.to(device)
    model.eval()
    X, _ = unbatch(batch, device)
    predictions = model(X)
    # x.cpu() eturns a copy of this object in CPU memory.
    # if this object is already in CPU memory and on the correct device,
    # then no copy is performed and the original object is returned.
    return [x.cpu() for x in X], predictions

def predict(model: fasterrcnn_resnet50_fpn,
            data_loader:torch.utils.data.DataLoader,
            device:str=None) -> tuple:
    """
    gets predictions for a batch of data
    - model: pretrained faster residual convolutional neural network
    - data_loader: data loader which contains the batch
    - device: string giving info if we gonna run this in a CPU or GPU
    Returns:
        - images: list of tensors of the images
        - predictions: list of dicts containing the predictions for
            the bounding boxes, labels, and confidence scores
    """
    images, predictions = [], []
    for batch in data_loader:
        X, p = predict_batch(batch, model, device)
        images.append(X)
        predictions.append(p)
    return images, predictions

def decode_prediction(prediction: dict,
                      score_threshold: float=.8,
                      nms_iou_threshold: float=.2) -> tuple:
    """
    decodes the prediction
    prediction: dictionary containing the predictions for
                the bounding boxes, labels, and confidence score
    score_thershold: filter predictions with lower score than the thershold
    nms_iou_treshold: intersection over union for non-max suppression.
                      discards overlapping bounding boxes
    returns tuple of filtered (boxes, scores, labels)
    """
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    if score_threshold:
        wanted = scores > score_threshold
        boxes = boxes[wanted]
        scores = scores[wanted]
        labels = labels[wanted]
    
    if nms_iou_threshold:
        wanted = torchvision.ops.nms(boxes=boxes, scores=scores,
                                    iou_threshold=nms_iou_threshold)
        boxes = boxes[wanted]
        scores = scores[wanted]
        labels = labels[wanted]
    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
