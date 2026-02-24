import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

# Import your custom modules
from models.meta_skin_next import FairMetaNext
from utils.dataloader import get_dataloaders


# ==========================================
# 1. HYBRID METRIC LEARNING LOSS FUNCTIONS
# ==========================================

class FocalLoss(nn.Module):
    """
    Penalizes the model for easy examples (like common nevi)
    and forces it to focus on hard, rare examples (like melanomas).
    """

    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights tensor

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply the Focal Loss modulating factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class CenterLoss(nn.Module):
    """
    Forces feature vectors of the same class to cluster tightly around a learned "Center".
    Crucial for Few-Shot Learning of rare diseases like Dermatofibroma.
    """

    def __init__(self, num_classes=8, feat_dim=256):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Learnable centers for each of the 8 classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        # Fetch the center for the corresponding true label
        centers_batch = self.centers.index_select(0, labels)

        # Calculate Squared Euclidean Distance between feature and its class center
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss


# ==========================================
# 2. MAIN TRAINING LOOP
# ==========================================

def train_model():
    # --- Configuration ---
    IMG_DIR = "../data/raw/ISIC_2019_Training_Input"
    GT_CSV = "../data/raw/ISIC_2019_Training_GroundTruth.csv"
    META_CSV = "../data/raw/ISIC_2019_Training_Metadata.csv"

    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    LAMBDA_WEIGHT = 0.05  # How much weight to give the Center Loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Setup Data ---
    train_loader, dataset = get_dataloaders(IMG_DIR, GT_CSV, META_CSV, batch_size=BATCH_SIZE)
    print(f"Total training images: {len(dataset)}")

    # --- Setup Architecture ---
    model = FairMetaNext(num_classes=8, meta_dim=10).to(device)

    # --- Setup Losses ---
    criterion_focal = FocalLoss(gamma=2.0).to(device)
    criterion_center = CenterLoss(num_classes=8, feat_dim=256).to(device)

    # --- Setup Optimizers ---
    # We need TWO optimizers: One for the Neural Network, one for the Center Prototypes
    optimizer_model = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, meta, labels in progress_bar:
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)

            # Zero gradients
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            # Forward Pass: We get logits (for Focal Loss) and features (for Center Loss)
            logits, features, attn_weights = model(images, meta)

            # Calculate Hybrid Loss
            loss_focal = criterion_focal(logits, labels)
            loss_center = criterion_center(features, labels)

            # L_total = L_focal + (lambda * L_center)
            loss_total = loss_focal + (LAMBDA_WEIGHT * loss_center)

            # Backward Pass
            loss_total.backward()

            # Step the Model Optimizer
            optimizer_model.step()

            # Step the Center Loss Optimizer (requires a special learning rate adjustment)
            for param in criterion_center.parameters():
                param.grad.data *= (1. / LAMBDA_WEIGHT)
            optimizer_center.step()

            # Metrics Tracking
            running_loss += loss_total.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update Progress Bar
            progress_bar.set_postfix({
                'Loss': f"{loss_total.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%"
            })

        # Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"--- Epoch [{epoch + 1}/{EPOCHS}] Summary | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% ---")

        # Save Checkpoint
        os.makedirs("../checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"../checkpoints/fairmeta_next_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    train_model()