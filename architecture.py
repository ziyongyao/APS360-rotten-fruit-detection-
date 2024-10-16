import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path (encoder)
        self.encoder1 = self.conv_block(1, 64)  # Input to 64 channels
        self.encoder2 = self.conv_block(64, 128)  # Encoder block
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)  # Bottleneck with 256 channels

        # Expanding path (decoder)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Upsampling
        self.decoder1 = self.conv_block(256, 128)  # Decoder block (128 channels)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsampling
        self.decoder2 = self.conv_block(128, 64)  # Decoder block (64 channels)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.maxpool(e1)
        e2 = self.encoder2(p1)
        p2 = self.maxpool(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder path
        d1 = self.upconv1(b)
        d1 = torch.cat((e2, d1), dim=1)  # Skip connection
        d1 = self.decoder1(d1)
        d2 = self.upconv2(d1)
        d2 = torch.cat((e1, d2), dim=1)  # Skip connection
        d2 = self.decoder2(d2)

        # Final output layer
        out = self.final_conv(d2)
        return out

# Example usage
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(8, 1, 224, 224)  # (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)
