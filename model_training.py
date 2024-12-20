import help_fc
import architecture
import architecture_up
import architecture_upv3
import arch_base
import train
import torch
import torchvision
import torchvision.transforms as transforms
import os

# Training the model
"""if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = architecture.UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train.train_general(model, train_dataset, val_dataset, batch_size=32, learning_rate=1e-3, num_epochs=20)

    # Create the directory if it does not exist
    os.makedirs('./saved_models', exist_ok=True)

    # Save the trained model
    model_path = './saved_models/' + help_fc.get_model_name("UNet", batch_size=32, learning_rate=1e-3, epoch=20)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")"""

# This code is to test the correctness of each function
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create a very small dataset subset (about 50 samples) to test the correctness of each function
    train_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/val', transform=transform)

    # Reduce the dataset size for testing (subset of 50 samples)
    small_train_dataset, _ = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset) - 1000])
    small_val_dataset, _ = torch.utils.data.random_split(val_dataset, [250, len(val_dataset) - 250])

    train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(small_val_dataset, batch_size=16, shuffle=False)

    model = architecture_upv3.UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Train the model on the small subset
    train.train_general(model, small_train_dataset, small_val_dataset, batch_size=16, learning_rate=1e-3, num_epochs=20)

    # Create the directory if it does not exist
    os.makedirs('./saved_models', exist_ok=True)

    # Save the trained model
    model_path = './saved_models/' + help_fc.get_model_name("test_UNet_upv3", batch_size=16, learning_rate=1e-3, epoch=20)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# train the baseline model
'''if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root='./new-small-dataset/val', transform=transform)

    # Reduce the dataset size for testing (subset of 50 samples)
    small_train_dataset, _ = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset) - 1000])
    small_val_dataset, _ = torch.utils.data.random_split(val_dataset, [250, len(val_dataset) - 250])

    train_loader = torch.utils.data.DataLoader(small_train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(small_val_dataset, batch_size=16, shuffle=False)

    baseline_model = arch_base.SimpleCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_model = baseline_model.to(device)

    train.train_general(baseline_model, small_train_dataset, small_val_dataset, batch_size=16, learning_rate=5e-4, num_epochs=20)

    # Create the directory if it does not exist
    os.makedirs('./saved_models', exist_ok=True)

    # Save the trained model
    model_path = './saved_models/' + help_fc.get_model_name("baseline_CNN", batch_size=16, learning_rate=5e-4, epoch=20)
    torch.save(baseline_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")'''