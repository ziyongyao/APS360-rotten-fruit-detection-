import torch
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path
def get_accuracy(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            outputs = model(imgs)
            probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            predictions = (probabilities >= 0.5).float()  # Threshold at 0.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    model.train()  # Set back to training mode
    return correct / total