import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from sklearn.model_selection import train_test_split

# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
MAX_EPOCHES = 3

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


def get_model(model_path):
    model = torch.load(model_path)
    # print(torch.load(model_path))
    # exit()
    # model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    # model = eval(model_class)(input_shape, num_actions).to(device)
    # model.load_state_dict(model_state_dict)
    return model


def save_model(model, model_path='./model.pt'):
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

def load_data(data_path='./data/train.csv'):
    df = pd.read_csv("./data/train.csv")
    y = df['label'].values
    X = df.drop(['label'], 1).values
    print('X.shape = ', X.shape)
    print('y.shape = ', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    torch_X_train = torch.from_numpy(X_train).view(-1,1,28,28).type(torch.FloatTensor).to(device)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
    torch_X_test = torch.from_numpy(X_test).view(-1,1,28,28).type(torch.FloatTensor).to(device)
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
    train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train(model_class, train_loader,model_path=False):
    if model_path != False:
        model = get_model(model_path)
    else:
        model = model_class()
    print(model)
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model.parameters())
    print(opt)
    loss_count = []
    for epoch in range(MAX_EPOCHES):
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            batch_x = x.to(device)  # torch.Size([128, 1, 28, 28])
            batch_y = y.to(device)  # torch.Size([128])
            out = model.forward(batch_x)  # torch.Size([128,10])
            loss = loss_func(out, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(out.data.argmax(1), batch_y, end=' ')
            # exit()
            correct += (out.data.argmax(1) == batch_y).sum()
            if i % 20 == 0:
                loss_count.append(loss)
                print('Episode: {}\tLoss: {:.6f}\tAccurancy: {:.3f}%'\
                      .format(i, loss.data, float(correct*100)/float((i+1) * BATCH_SIZE)))
                torch.save(model, './model.pt')

            # if i % 100 == 0:
            #     for a, b in test_loader:
            #         test_x = a
            #         test_y = b
            #         out = model.forward(test_x)
            #         # print('test_out:\t',torch.max(out,1)[1])
            #         # print('test_y:\t',test_y)
            #         accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
            #         print('accuracy:\t', accuracy.mean())
            #         break

def test(test_loader, model_path):
    model = get_model(model_path)
    correct = 0
    for i, (x, y) in enumerate(test_loader):
        batch_x = x.to(device)
        batch_y = y.to(device)
        out = model.forward(batch_x)
        correct += (out.data.argmax(1) == batch_y).sum()
    print('Test Accurancy: {:.3f}'.format(float(correct * 100) / float(len(test_loader) * BATCH_SIZE)))

if __name__ == '__main__':
    is_train = True
    train_loader, test_loader = load_data()

    if is_train:
        # train(ConvNet, train_loader, model_path='./model.pt')
        train(ConvNet, train_loader)

    test(test_loader, './model.pt')

