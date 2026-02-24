FairMeta-Next: Context-Guided Dermatological AI 🔬

FairMeta-Next is an advanced, multi-modal deep learning architecture for skin cancer triage. It bridges the "Double Domain Shift" in dermatological AI (Dermoscopy vs. Clinical Photography, and Light vs. Dark Skin) by replacing blind data concatenation with Context-Guided Cross-Attention Fusion and Evidential Deep Learning.

This repository contains the official PyTorch implementation for the FairMeta-Next PhD research project, optimized for highly imbalanced 8-class taxonomies (ISIC 2019).

✨ Core Architectural Innovations

Unlike standard medical CNNs that simply append patient metadata to the end of an image network, FairMeta-Next utilizes a highly dynamic, spatially-aware architecture:

The Eye (ConvNeXt-Tiny Backbone): Replaces standard ResNets with a hierarchical ConvNeXt backbone, utilizing large $7\times7$ depth-wise convolutions to capture global lesion asymmetry and network patterns.

The Brain (Multi-Head Cross-Attention): Patient metadata (Age, Sex, Anatomical Site) acts as a mathematical "Query" to spatially scan the image. It actively attends to diagnostic pixels while filtering out visual noise (hair, glare, low contrast on dark skin).

The Penalty (Anti-Spurious Masking): Implements a spatial penalty mask that forces the attention mechanism to ignore surgical ink, colorful rulers, and dark image borders, solving the "Clever Hans" effect.

The Confidence (Evidential Deep Learning): Replaces the standard Softmax layer with a Dirichlet Evidential output, allowing the model to calculate mathematical Uncertainty. The AI can now safely say "I don't know" when faced with out-of-distribution artifacts.

The Clusters (Hybrid Metric Learning): Combines Evidential Focal Loss with Center Loss to tightly cluster feature vectors in the latent space, enabling Few-Shot Learning for extremely rare diseases (like Dermatofibroma).

📊 Dataset & Taxonomy

This model is trained on the ISIC 2019 Archive and evaluated against Fitzpatrick17k diversity metrics. It uses a unified 8-class clinical taxonomy:

Class

Diagnosis

Class

Diagnosis

0

Melanoma (MEL)

4

Benign Keratosis (BKL)

1

Melanocytic Nevus (NV)

5

Dermatofibroma (DF)

2

Basal Cell Carcinoma (BCC)

6

Vascular Lesion (VASC)

3

Actinic Keratosis (AK)

7

Squamous Cell Carcinoma (SCC)

Note: Missing metadata is handled gracefully via a Maximum Entropy (Neutral) Imputation strategy ($0.5$), ensuring the Cross-Attention module does not hallucinate demographic bias.

🚀 Installation & Setup

Prerequisites

Python 3.9+

PyTorch 2.0+ (with CUDA support)

An NVIDIA GPU (Scripts are pre-optimized for 6GB VRAM GPUs like the GTX 1660 Ti using Automatic Mixed Precision).

Clone the Repository

git clone [https://github.com/YOUR-USERNAME/FairMeta-Next.git](https://github.com/YOUR-USERNAME/FairMeta-Next.git)
cd FairMeta-Next


Install Dependencies

pip install torch torchvision pandas numpy Pillow tqdm


💻 Running the Code

1. Prepare the Data

Ensure your ISIC 2019 datasets are extracted into the data/raw/ directory:

data/raw/ISIC_2019_Training_Input/ (Images)

data/raw/ISIC_2019_Training_GroundTruth.csv

data/raw/ISIC_2019_Training_Metadata.csv

2. Start Training (Local PC / 6GB VRAM)

The local training script is heavily optimized with torch.amp.autocast, pin_memory=True, and prefetch_factor=2 to prevent GPU starvation and CUDA Out-Of-Memory errors on standard laptops.

python src/train_local_pc.py


📁 Repository Structure

FairMeta-Next/
├── data/                  # Ignored in Git (Place your ISIC images/CSVs here)
├── src/
│   ├── models/
│   │   └── meta_skin_next.py   # ConvNeXt + Cross-Attention Architecture
│   ├── utils/
│   │   └── dataloader.py       # Multi-modal Dataset loader & One-Hot Encoder
│   ├── train_local_pc.py       # Hardware-optimized training loop (AMP/Center Loss)
│   └── test_run.py             # Pipeline validation script
├── checkpoints/           # Saved .pth weights
└── README.md


📝 License & Citation

This project is part of an ongoing PhD Research initiative. If you utilize this architecture or the hybrid loss functions in your own work, please cite accordingly.

Code released under the MIT License.