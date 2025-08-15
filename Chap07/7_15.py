import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import matplotlib.pyplot as plt

# define input size, hidden layer size, output size
D_i, D_k, D_o = 10, 40, 5

# create model with two hidden layers
model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

# He initialization
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model.apply(weights_init)

# choose least squares loss criterion
criterion = nn.MSELoss()

# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)

# object that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# create 100 random data points and store in data loader class
x = torch.randn(100, D_i)
y = torch.randn(100, D_o)
data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)

losses = [0] * 100

# loop over the dataset 100 times
for epoch in tqdm(range(100)):
    epoch_loss = 0.0

    # loop over batches
    for i, data in enumerate(data_loader):

        x_batch, y_batch = data # retrieve inputs and labels for this batch
        optimizer.zero_grad() # zero the parameter gradients

        # forward pass
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        loss.backward() # backward pass

        optimizer.step() # SGD update

        epoch_loss += loss.item() # update statistics
    
    losses[epoch] = epoch_loss

    # print error
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')

    # tell scheduler to consider updating learning rate
    scheduler.step()

plt.plot([epoch for epoch in range(1, 101)], losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()