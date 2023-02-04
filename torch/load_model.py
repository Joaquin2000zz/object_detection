"""
module which contains prepare_model function
"""
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def prepare_model(num_classes=2, feature_extraction=True) -> fasterrcnn_resnet50_fpn:
    """
    prepares a pretrained faster residual convolutional neural network model
    which classify in between a range of 91 classes
    - num_classes: classes needed during our training.
        2 is a binary classificator
    - feature_extraction: boolean balue to determine if
        freeze pre trained weights avoiding
        the weights updating throughout gradients
    Returns:
        - model: the prepared model
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    if feature_extraction:
        for p in model.parameters():
            p.requires_grad = False
    # replacing for the required number of classes
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=input_features,
                                                      num_classes=num_classes)
    return model
