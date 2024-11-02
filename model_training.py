import help_fc
import architecture
import train
import torch
import torchvision
import torchvision.transforms as transforms

# Training the model
if __name__ == "__main__":
    # Assuming train_dataset and val_dataset are defined and loaded properly
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='/content/new-fruits-dataset/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root='/content/new-fruits-dataset/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = architecture.UNet()
    train.train_general(model, train_dataset, val_dataset, batch_size=32, learning_rate=1e-4, num_epochs=1)

    # Save the trained model
    model_path = help_fc.get_model_name(model.name, batch_size=32, learning_rate=1e-4, epoch=1, use_coda=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")