import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

# Import custom modules
from models.meta_skin_next import FairMetaNext
from utils.dataloader import get_dataloaders


# ==========================================
# HYBRID METRIC LEARNING LOSS FUNCTIONS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes=8, feat_dim=256):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss


# ==========================================
# MAIN TRAINING LOOP (GENERAL)
# ==========================================
def train_model():
    # --- Configuration ---
    IMG_DIR = "../data/raw/ISIC_2019_Training_Input"
    GT_CSV = "../data/raw/ISIC_2019_Training_GroundTruth.csv"
    META_CSV = "../data/raw/ISIC_2019_Training_Metadata.csv"

    # Standard Deep Learning Configuration
    BATCH_SIZE = 32  # Standard batch size for high-end GPUs (>12GB VRAM)
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    LAMBDA_WEIGHT = 0.05

    # Generic Hardware Detection (Supports NVIDIA, Apple Silicon Mac, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimize NVIDIA convolutions
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple M1/M2/M3 chips
    else:
        device = torch.device("cpu")

    print(f"Training on device: {device} (General Configuration)")

    # --- Setup Data ---
    train_loader, dataset = get_dataloaders(IMG_DIR, GT_CSV, META_CSV, batch_size=BATCH_SIZE)
    print(f"Total training images: {len(dataset)}")

    # --- Setup Architecture ---
    model = FairMetaNext(num_classes=8, meta_dim=10).to(device)

    # --- Setup Losses & Optimizers ---
    criterion_focal = FocalLoss(gamma=2.0).to(device)
    criterion_center = CenterLoss(num_classes=8, feat_dim=256).to(device)

    optimizer_model = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

    # Optional AMP scaler for high-end NVIDIA GPUs (improves speed on RTX 3000/4000 series)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, meta, labels in progress_bar:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)

            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            # Forward pass (with optional mixed precision for standard speedups)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, features, attn_weights = model(images, meta)
                    loss_focal = criterion_focal(logits, labels)
                    loss_center = criterion_center(features, labels)
                    loss_total = loss_focal + (LAMBDA_WEIGHT * loss_center)

                scaler.scale(loss_total).backward()
                scaler.step(optimizer_model)
                scaler.step(optimizer_center)
                scaler.update()
            else:
                # Standard precision fallback for CPU / Mac MPS
                logits, features, attn_weights = model(images, meta)
                loss_focal = criterion_focal(logits, labels)
                loss_center = criterion_center(features, labels)
                loss_total = loss_focal + (LAMBDA_WEIGHT * loss_center)

                loss_total.backward()
                optimizer_model.step()
                optimizer_center.step()

            # Adjust center loss learning rate manually after step
            for param in criterion_center.parameters():
                param.grad.data *= (1. / LAMBDA_WEIGHT)

            # Metrics Tracking
            running_loss += loss_total.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{loss_total.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%"
            })

        # Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"--- Epoch [{epoch + 1}/{EPOCHS}] Summary | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% ---")

        os.makedirs("../checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"../checkpoints/fairmeta_next_epoch_{epoch + 1}_general.pth")


if __name__ == '__main__':
    train_model()