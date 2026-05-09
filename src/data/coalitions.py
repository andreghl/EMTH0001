import h5py
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


class CoalitionNN(nn.Module):

    """
    Neural network to approximate the value of the expected collaborative gain based on a VRP instance and a possible coalition of vehicles.
    """

    def __init__(self):
        super().__init__()

        self.DmNet = nn.Sequential(
            nn.Linear(48, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.coalNET = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, xDm, xC):

        xDmNet = self.DmNet(xDm)
        xCNet = self.coalNET(xC)
        x = torch.cat([xDmNet, xCNet], dim = 1)
        return self.head(x)


if __name__ == '__main__':

    with h5py.File("instances.h5", "r") as f:
        print(list(f.keys()))

        Dm = f['Dm'][:, :, 1:]
        A = f['assign'][:]
        C = f['coalitions'][:]
        v = f['v'][:]

        # Get tensor for input of NN
        Dm = torch.from_numpy(Dm)
        A = torch.from_numpy(A)
        C = torch.from_numpy(C)
        v = torch.from_numpy(v)

    """
    N = 9 (number of customers)
    D = 3 (number of depots)
    N + D = 12 (number of nodes; sum of customers and depots)
    K = 8 = 2 ** D (number of coalitions)   
    print(Dm.shape)
    > (10000, 12, 4)
    print(A.shape)
    > (10000, 12, 8)
    print(C.shape)
    > (10000, 8, 3)
    print(v.shape)
    > (10000, 8)
    """

    # Train a model to predict the expected collaboration gain per coalition so need one instance per coalition and take one specific coalition as input to get its own collaboration as output.
    xDm = Dm.unsqueeze(1).repeat(1, 8, 1, 1)
    xDm = xDm.reshape(-1, 12 * 4)
    xC = C.reshape(-1, 3)
    y = v.reshape(-1, 1)

    """
    print(xDm.shape)
    > (80000, 48)
    print(xC.shape)
    > (80000, 3)
    print(y.shape)
    > (80000, 1)
    """

    # NN Model Training
    N = xDm.shape[0]
    split = int(0.8 * N)

    train = TensorDataset(xDm[:split], xC[:split], y[:split])
    test = TensorDataset(xDm[split:], xC[split:], y[split:])
    test = DataLoader(test, batch_size = 256, shuffle = True)
    # TODO: check size of test dataset

    data = TensorDataset(xDm.float(), xC.float(), y.float())
    batches = DataLoader(train, batch_size = 256, shuffle = True)

    model = CoalitionNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    # TODO: Need to keep logs of the gradient to see what is going
    # TODO: Need to initialize the NN model

    epochs = 100

    for epoch in range(epochs):

        totalLoss = 0
        val_loss = 0

        for (bDm, bC, bY), (tDm, tC, tY) in zip(batches, test):

            learn = model(bDm, bC)
            loss = criterion(learn, bY)

            val = model(tDm, tC)
            val_loss += criterion(val, tY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}, Training Loss: {totalLoss:.4f}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "coalitions.pth")

'''
motivation for CoalitionNN:
> Whilst this is not the exact task agents must perform in the collaborative routing scenario, the intuition is that the neural network should still learn useful patterns which are transferable to the full collaborative routing problem.

> we want to know when we add this NN to gym agent whether NN or agent is learning/not learning (to 
'''

# TODO: remove repo from OneDrive
# TODO: use and look at the Gym NN agent (after chap 7 or 8) {gdrl}