import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CalibModel
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import CalibDataset



torch.manual_seed(2022)


def save_loss_plot(train, test):

    plt.plot(train, label = 'train loss', color='green')
    plt.plot(test, label = 'test loss', color='red')
    
    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    plt.legend()

    plt.savefig('loss.png')
    
    plt.close('all')


MAX_EPOCHS = 50
LR = 1e-4


net = CalibModel()
lossfn =  nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)


traindataset = CalibDataset(train=True)
testdataset = CalibDataset(train=False)

traindataloader = DataLoader(traindataset, batch_size=64, shuffle=True)
testdataloader = DataLoader(testdataset, batch_size=64, shuffle=True)


train_loss_over_time = []
test_loss_over_time = []


lowest_test_loss = float('inf')

for epoch in tqdm(range(MAX_EPOCHS)):

    train_loss_epoch = []
    test_loss_epoch = []


    net.train()

    for x,y in traindataloader:
        optimizer.zero_grad()
        
        pred = net(x)
        loss = lossfn(pred,y)

        train_loss_epoch.append(loss.item())

        loss.backward()
        optimizer.step()


    net.eval()

    with torch.no_grad():
        for x,y in testdataloader:

            pred = net(x)
            loss = lossfn(pred,y)

            test_loss_epoch.append(loss.item())


    train_loss_epoch = sum(train_loss_epoch)/len(train_loss_epoch)
    test_loss_epoch = sum(test_loss_epoch)/len(test_loss_epoch)


    train_loss_over_time.append(train_loss_epoch)
    test_loss_over_time.append(test_loss_epoch)

    save_loss_plot(train_loss_over_time, test_loss_over_time)


    print(f'EPOCH : {epoch} \ Train Loss : {train_loss_epoch:.3f} \ Test Loss : {test_loss_epoch:.3f}')
    
    if test_loss_epoch < lowest_test_loss:

        torch.save(net.state_dict(),'model.pt')

        if lowest_test_loss != float('inf'):
            print(f'---------- {(test_loss_epoch-lowest_test_loss)/lowest_test_loss * -100 :.2f} % IMPROVEMENT ----------')

        print('---------- SAVING CHECKPOINT ----------')


        lowest_test_loss = test_loss_epoch