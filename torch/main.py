#!/usr/bin/env python3
"""
main module, starts the training and the test
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
train = __import__('train_model').train
predict = __import__('predict_and_test').predict
decode_prediction = __import__('predict_and_test').decode_prediction


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_aviable() else torch.device('cpu')

    labels = ['QR']
    label_dict = {l:i for l, i in enumerate(labels)}
    reverse_label_dict = {v:k for k, v in label_dict.values()}

    log, val_dl, model = train(device=device, labels=labels)
    images, predictions = predict(val_dl, model, device)
    boxes, scores, labels = decode_prediction(predictions)

    img_index = 0
    fig, ax = plt.subplots(fig_size=[5, 5])
    ax.imshow(images[img_index].permute(1, 2, 0).numpy())

    for i, b in enumerate(boxes):
        rect = patches.Rectangle(b[:2].astype(int),
                                 (b[2] - b[0]).astype(int),
                                 (b[3] - b[1]).astype(int),
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor=None)
        ax.add_patch(rect)
        ax.text(b[0], b[1] - 5,
                f"{reverse_label_dict[labels[i]]}: {scores[i]}",
                color='r')
    plt.show()
