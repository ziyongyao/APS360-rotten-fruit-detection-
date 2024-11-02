import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import help_fc
import architecture
import matplotlib.pyplot as plt
def train_general(model, train_dataset, val_dataset, batch_size=32, learning_rate=1e-4, num_epochs=20, use_cuda=True):
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)

    # If CUDA is available and use_cuda is True, move model to GPU
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # standard procedure for starting, using SGD for good performance
    criterion =  nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    iters, losses, train_acc, val_acc = [], [], [], []
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    model.train()
    index = 0
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU if available
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the  loss using cross entropy
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # Save the current training information
            iters.append(index)
            losses.append(float(loss)) 
            index+= 1
        train_acc.append(help_fc.get_accuracy(model,train_loader)) # compute training accuracy 
        val_acc.append(help_fc.get_accuracy(model, val_loader))  # compute validation accuracy
        print("Epoch",epoch,"training accuracy",train_acc[-1])
        print("Epoch",epoch,"validation accuracy",val_acc[-1])
        # Save the current model (checkpoint) to a file
        if epoch%20 == 0:
            model_path = help_fc.get_model_name(model.name, batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)
    model_path = help_fc.get_model_name(model.name, batch_size, learning_rate, epoch)
    torch.save(model.state_dict(), model_path)
    print('Stop!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("note,training time is ",elapsed_time)

    # Plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training Curve")
    plt.plot(range(1 ,num_epochs+1), train_acc, label="Train")
    plt.plot(range(1 ,num_epochs+1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
