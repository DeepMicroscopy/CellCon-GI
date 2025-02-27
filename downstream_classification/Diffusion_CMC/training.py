import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import os
from PIL import Image
import logging
from datetime import datetime

# Setup logging
log_file = f"diffusion_cmc_img_cmc_mask_amibr.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet default size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load data from folders
def load_data_from_folder(folder_path, classes):
    images = []
    labels = []
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        class_folder = os.path.join(folder_path, cls_name)
        if not os.path.isdir(class_folder):
            continue
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if os.path.isfile(file_path):
                images.append(file_path)
                labels.append(class_to_idx[cls_name])

    return images, labels

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Specify the classes
classes = ['Atypical', 'Normal']

# Load dataset
images, labels = load_data_from_folder(
    '/home/MICCAI25/Amibr_diffusion_synthetic_train_set/CMCmodel_CMCdataset', 
    classes
)

# Define the EfficientNet model for binary classification
class BinaryEfficientNet(nn.Module):
    def __init__(self):
        super(BinaryEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Sequential(
            nn.Linear(self.model._fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training configuration
num_epochs = 100
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCELoss()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold Cross Validation
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
    logger.info(f"Starting Fold {fold + 1}")

    # Split the raw lists into training and validation parts
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Create separate Datasets with different transforms
    train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)

    # Calculate class weights for WeightedRandomSampler
    class_counts = [0] * len(classes)
    for lbl in train_labels:
        class_counts[lbl] += 1
    class_weights = [1.0 / count for count in class_counts]

    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize model, optimizer, and scheduler
    model = BinaryEfficientNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_balanced_accuracy = 0.0
    best_model_path = f'diffusion_cmc_img_cmc_mask_amibr_fold{fold + 1}_best.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        # Training loop
        for images_batch, labels_batch in tqdm(train_loader, 
                                               desc=f"Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs} - Training"):
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            labels_batch = labels_batch.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())

        train_balanced_accuracy = balanced_accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        # Validation loop
        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, 
                                                   desc=f"Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs} - Validation"):
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                labels_batch = labels_batch.float().unsqueeze(1)

                outputs = model(images_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())

        val_balanced_accuracy = balanced_accuracy_score(val_targets, val_preds)

        if val_balanced_accuracy > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

        # Logging metrics
        log_entry = (
            f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Balanced Accuracy: {train_balanced_accuracy:.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Balanced Accuracy: {val_balanced_accuracy:.4f}"
        )
        logger.info(log_entry)

    logger.info(f"Fold {fold + 1} - Best Validation Balanced Accuracy: {best_val_balanced_accuracy:.4f}")
    fold_accuracies.append(best_val_balanced_accuracy)

# Calculate and log overall performance
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
logger.info(f"Average Validation Balanced Accuracy across all folds: {average_accuracy:.4f}")
