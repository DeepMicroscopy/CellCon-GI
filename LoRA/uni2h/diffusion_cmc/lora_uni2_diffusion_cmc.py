import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from huggingface_hub import login
from timm import create_model
from peft import get_peft_model, LoraConfig
from timm.layers import SwiGLUPacked

# Logging setup
log_file = "uni2_lora_diffusion_cmc_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face login
login(token="your_huggingface_token_here")  # Replace with your Hugging Face token

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv", "proj", "fc1", "fc2"],
    lora_dropout=0.3,
    bias="none",
    modules_to_save=["head"]
)

# Image transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return image, label

# Load dataset from folder
def load_image_paths_from_folder(root_dir):
    image_paths = []
    labels = []
    label_map = {'Atypical': 0, 'Normal': 1}
    for label_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir) or label_name not in label_map:
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label_map[label_name])
    return image_paths, labels

# Data
root_data_dir = '/data/MELBA-AmiBr/TriCon-GI/downstream_classification/Amibr_diffusion_synthetic_train_set/CMCmodel_CMCdataset'
images, labels = load_image_paths_from_folder(root_data_dir)

# Hyperparameters
batch_size = 8
num_epochs = 100
early_stop_patience = 15
criterion = nn.BCEWithLogitsLoss()
fold_accuracies = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
    logger.info(f"Starting Fold {fold + 1}")

    base_model = create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        embed_dim=1536,
        init_values=1e-5,
        mlp_ratio=2.66667 * 2,
        num_classes=1,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True
    )
    base_model = base_model.to(device)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ImageDataset([images[i] for i in train_idx], [labels[i] for i in train_idx], train_transform)
    val_dataset = ImageDataset([images[i] for i in val_idx], [labels[i] for i in val_idx], val_transform)

    train_labels = [labels[i] for i in train_idx]
    class_counts = [train_labels.count(i) for i in range(2)]
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7, verbose=True)

    best_val_acc = 0.0
    best_model_path = f'uni2_lora_diffusion_cmc_fold_{fold + 1}_best.pth'
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images_batch, labels_batch in tqdm(train_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Training"):
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images_batch)
            if outputs.ndim == 3:
                outputs = outputs[:, 0]
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            elif outputs.ndim == 2 and outputs.size(1) != 1:
                outputs = outputs[:, :1]

            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())

        train_bal_acc = balanced_accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1} Validation"):
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                outputs = model(images_batch)
                if outputs.ndim == 3:
                    outputs = outputs[:, 0]
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(1)
                elif outputs.ndim == 2 and outputs.size(1) != 1:
                    outputs = outputs[:, :1]

                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())

        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        scheduler.step(val_bal_acc)

        logger.info(
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Bal Acc: {train_bal_acc:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Bal Acc: {val_bal_acc:.4f}"
        )

    logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_val_acc:.4f}")
    fold_accuracies.append(best_val_acc)
    torch.cuda.empty_cache()

# Final summary
avg_acc = sum(fold_accuracies) / len(fold_accuracies)
logger.info(f"Average Validation Balanced Accuracy across folds: {avg_acc:.4f}")
