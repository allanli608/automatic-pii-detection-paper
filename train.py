import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.unet_model import UNet
from src.dataset import SegmentationDataset

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = loader
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # Forward
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ]
    )

    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_ds = SegmentationDataset(
        image_dir="data/train_images/",
        mask_dir="data/train_masks/",
        transform=transform,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save checkpoint
        torch.save(model.state_dict(), "checkpoints/unet_checkpoint.pth")


if __name__ == "__main__":
    main()
