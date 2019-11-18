net = Net()
criterion = nn.BCELoss()  # Binary Cross Entropy
# SGD optimizer with learning rate 0.001 and momentum 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for batch in range(1000):
    # get mini-batch
    indices = np.random.choice(len(x_train), size=(30))
    inputs = x_train[indices]
    labels = y_train[indices]

    # format input into [#sample, #channel, #feature]
    inputs = torch.from_numpy(inputs).view([-1, 1, dim]).float()
    labels = torch.from_numpy(labels).view([-1, 1]).float()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs).view([-1, 1])
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
