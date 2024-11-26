import torch
import torch.nn as nn
import torch.optim as optim

# Define the UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder path (Downsampling)
        self.enc_conv1 = self.conv_block(3, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)
        self.enc_conv5 = self.conv_block(512, 1024)
        
        # Max Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder path (Upsampling)
        self.up_conv4 = self.up_conv(1024, 512)
        self.dec_conv4 = self.conv_block(1024, 512)
        
        self.up_conv3 = self.up_conv(512, 256)
        self.dec_conv3 = self.conv_block(512, 256)
        
        self.up_conv2 = self.up_conv(256, 128)
        self.dec_conv2 = self.conv_block(256, 128)
        
        self.up_conv1 = self.up_conv(128, 64)
        self.dec_conv1 = self.conv_block(128, 64)
        
        # Global Average Pooling to create a feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(64, 1)  # Outputting a single logit

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),  # Added BatchNorm
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)   # Added BatchNorm
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        enc4 = self.enc_conv4(self.pool(enc3))
        enc5 = self.enc_conv5(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.up_conv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec_conv4(dec4)
        
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)
        
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)
        
        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)

        # Global Average Pooling
        gap = self.global_avg_pool(dec1)
        gap = gap.view(gap.size(0), -1)  # Flatten the output

        # Fully connected layer for classification
        out = self.fc(gap).squeeze(1)
        
        return out


def train(model, criterion, optimizer, num_epochs=5):
    # Simple random data for testing
    batch_size = 8
    input_data = torch.randn(batch_size, 3, 224, 224)  # 3-channel images
    target = torch.randint(0, 2, (batch_size,)).float()  # Random binary target (0 or 1)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()

        # Print training stats
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# Main function to test the model's learning ability
if __name__ == "__main__":
    model = UNet()  # Initialize the model
    criterion = nn.BCEWithLogitsLoss()  # Loss function for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

    trained_model = train(model, criterion, optimizer, num_epochs=5)
    print("Training complete.")
