import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling layer to create a feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for binary classification
        self.fc = nn.Linear(128, 1)  # Outputting a single logit

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Fully connected layer for classification
        out = self.fc(x).squeeze(1)
        return out


# Example usage
if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(8, 3, 224, 224)  # (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)
