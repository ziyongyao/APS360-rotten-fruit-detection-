import torch
import torch.nn as nn
import torch.optim as optim
import torch.version
import torchvision
import torchvision.transforms as transforms
import time
import help_fc
import architecture
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def convert_labels_to_binary(labels):
    # Convert labels to binary: 0 for fresh, 1 for rotten
    return (labels >= 3).float()

def train_general(model, train_dataset, val_dataset, batch_size=32, learning_rate=1e-4, num_epochs=20):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    # standard procedure for starting, using SGD for good performance
    criterion =  nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    early_stopping = EarlyStopping(patience=2)

    iters, losses, train_acc, val_acc = [], [], [], []
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    model.train()
    index = 0
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU if available

            # Convert labels to binary for binary classification
            #print(f"Original Labels: {labels}")
            labels = convert_labels_to_binary(labels)
            #print(f"Binary Labels: {labels}")

            out = model(imgs)             # forward pass
            loss = criterion(out, labels.float()) # compute the  loss using cross entropy
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # Save the current training information
            iters.append(index)
            losses.append(float(loss))
            index+= 1
        train_acc.append(help_fc.get_accuracy(model, train_loader, device)) # compute training accuracy
        with torch.no_grad():
            val_loss = sum(criterion(model(imgs.to(device)), convert_labels_to_binary(labels).to(device).float()) for imgs, labels in val_loader) / len(val_loader)
            val_acc.append(help_fc.get_accuracy(model, val_loader, device))  # compute validation accuracy
        print("Epoch",epoch,"training accuracy",train_acc[-1])
        print("Epoch",epoch,"validation accuracy",val_acc[-1])
        # Save the current model (checkpoint) to a file
        if epoch % 20 == 0:
            model_path = help_fc.get_model_name("test_model", batch_size, learning_rate, epoch)
            torch.save(model.state_dict(), model_path)
        scheduler.step(val_loss)
        print(f"Learning rate after epoch {epoch + 1}: {scheduler.optimizer.param_groups[0]['lr']}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered at epoch", epoch)
            break
        torch.cuda.empty_cache()
    model_path = help_fc.get_model_name("test_model", batch_size, learning_rate, epoch)
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

    completed_epochs = len(train_acc)
    plt.title("Training Curve")
    plt.plot(range(1, completed_epochs + 1), train_acc, label="Train")
    plt.plot(range(1, completed_epochs + 1), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
# confirm if CUDA is accessible for training
"""print(torch.version)
print(torch.version.cuda)  # Should print a version number like '12.4' if CUDA is installed
print(torch.cuda.is_available())  # Should return True if CUDA is properly set up
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")"""
