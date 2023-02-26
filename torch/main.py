#!/usr/bin/env python3
"""
main module, starts the training and the test
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
train = __import__('train_model').train
predict = __import__('predict_and_test').predict
decode_prediction = __import__('predict_and_test').decode_prediction
change_ext = __import__('change_ext').change_ext
prepare_loaders = __import__('prepare_loaders').prepare_loaders


if __name__ == '__main__':
    os.chdir('torch')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    change_ext(ext='.png')
    labels = ['QR']
    label2target = {l:i + 1 for i, l in enumerate(labels)}
    label2target['background'] = 0
    target2label = {v:k for k, v in label2target.items()}
    num_classes = len(target2label)

    if not os.path.isfile('model.pt'):
        log, test_dl, model = train(num_classes=num_classes,
                                    device=device, labels=label2target)
        torch.save(model.state_dict(), 'model.pt')
    else:
        model = fasterrcnn_resnet50_fpn()
        for p in model.parameters():
            p.requires_grad = False
        # replacing for the required number of classes
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=input_features,
                                                          num_classes=num_classes)
        model.load_state_dict(torch.load('model.pt'))
        train_dl, val_dl, test_dl = prepare_loaders('./', labels=label2target,
                                                    batch_size=4)
    images, predictions = predict(model, test_dl, device)

    for img_index, p in enumerate(predictions):
        print("antes del decode", p, img_index)
        boxes, scores, labels = decode_prediction(p)
        print('desp del decode')
        fig, ax = plt.subplots(figsize=[5, 5])
        print('desp del subplot')
        ax.imshow(images[img_index].permute(1, 2, 0).numpy())
        print("boxes:", boxes, "\nscores:", scores, "\nlabels:", labels)
        for i, b in enumerate(boxes):
            rect = patches.Rectangle(b[:2].astype(int),
                                     (b[2] - b[0]).astype(int),
                                     (b[3] - b[1]).astype(int),
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor=None)
            ax.add_patch(rect)
            ax.text(b[0], b[1] - 5,
                    f"{target2label[labels[i]]}: {scores[i]}",
                    color='r')
    plt.show()