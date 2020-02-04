import os
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Dataset import Dataset
from Models import CnnBasic, CnnReLU, CnnLeaky
from Training import train_model


def save_histograms(model, writer, step, l1, l2, l3, l4):
    create_histograms(model, writer, step, l1)
    create_histograms(model, writer, step, l2)
    create_histograms(model, writer, step, l3)
    create_histograms(model, writer, step, l4)


def create_histograms(model, writer, step, number):
    writer.add_histogram('Train/Conv{} - bias'.format(str(number)), model.sequential[number].bias, step)
    writer.add_histogram('Train/Conv{} - weight'.format(str(number)), model.sequential[number].weight, step)
    writer.add_histogram('Train/Conv{} - weight grad'.format(str(number)), model.sequential[number].weight.grad, step)


def run(use_gpu, train_dir, valid_dir, task, part, percent, model_type, learning_rate, patience, logdir, epochs):
    log_folder = './logs/' + task + '-' + str(percent) + '%-' + str(epochs) + 'epochs' + '/'
    log_dir = log_folder + logdir + '/'
    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    train_data = Dataset(use_gpu, train_dir, task, part, percent)
    valid_data = Dataset(use_gpu, valid_dir, task, part, percent)

    train_loader = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=1, shuffle=False)

    if model_type == 'CnnBasic':
        model = CnnBasic()
    if model_type == 'CnnReLU':
        model = CnnReLU()
    if model_type == 'CnnLeaky':
        model = CnnLeaky()

    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    optimizer = Adam(model.parameters(), learning_rate, weight_decay=.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=.3, verbose=False)

    for epoch in range(epochs):
        print('Epoch:', epoch + 1)
        train_loss, train_auc = train_model(model, train_loader, writer, epoch, True, optimizer)
        val_loss, val_auc = train_model(model, valid_loader, writer, epoch, False)
        scheduler.step(val_loss)

        if model_type == 'CnnBasic':
            save_histograms(model, writer, epoch + 1, 0, 3, 6, 9)
        else:
            save_histograms(model, writer, epoch + 1, 0, 5, 10, 15)
