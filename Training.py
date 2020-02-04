import torch
import numpy as np
from sklearn import metrics
from torch.autograd import Variable


def train_model(model, loader, writer, epoch, train, optimizer=None):
    preds = []
    labels = []
    losses = []

    if train:
        model.train()
    else:
        model.eval()

    for batch, (axial, sagittal, coronal, label, weight) in enumerate(loader):
        if loader.dataset.use_gpu:
            axial, sagittal, coronal, label = axial.cuda(), sagittal.cuda(), coronal.cuda(), label.cuda()
        axial, sagittal, coronal, label = Variable(axial), Variable(sagittal), Variable(coronal), Variable(label)

        if train:
            optimizer.zero_grad()

        prediction = model.forward(axial.float(), sagittal.float(), coronal.float())
        loss = loader.dataset.weighted_loss(prediction, label)
        losses.append(loss.item())

        pred = torch.sigmoid(prediction)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]
        preds.append(pred_npy)
        labels.append(label_npy)

        mode = "Valid"
        if train:
            loss.backward()
            optimizer.step()
            mode = "Train"

        false_positive, true_positive, threshold = metrics.roc_curve(labels, preds)
        auc = metrics.auc(false_positive, true_positive)

        if batch % 10 == 9:
            print(mode + '  Batch:', batch + 1, 'Loss:', np.round(np.mean(losses), 4), 'AUC:', np.round(auc, 4))
            writer.add_scalar(mode + '/Loss', np.mean(losses), epoch * len(loader) + batch + 1)
            writer.add_scalar(mode + '/Accuracy', auc, epoch * len(loader) + batch + 1)

    writer.add_scalar(mode + '/Accuracy epoch', auc, epoch + batch)

    avg_loss = np.round(np.mean(losses), 4)
    auc = np.round(auc, 4)

    return avg_loss, auc