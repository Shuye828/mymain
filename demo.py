import torch

from MedTS_TTT import MedTSTTT


def main():
    # Input shape: [batch, channels, time]
    x = torch.randn(8, 12, 1000)

    model = MedTSTTT(
        dim=128,
        max_channel=128,
        num_heads=8,
        num_layers=6,
        patch_size=8,
        num_classes=5,
    )

    logits = model(x)
    print("input:", x.shape)
    print("logits:", logits.shape)


if __name__ == "__main__":
    main()
