#!/usr/bin/env python3
"""ResNet-50 binary CT classification training/evaluation.

Adapted from Seyed Vala Khorasani, "Lung Cancer Detection Using Pretrained
ResNet-50" (Kaggle). Original notebook attribution and Apache 2.0 terms
are documented in README.md and LICENSE.

Project-specific dataset selection and reported experimental results are
explained in README.md. This script creates its OWN stratified 70/15/15
split from the images under dataset_root; it does not reproduce the paper's
separate 364-image testing collection by itself.
"""

import os
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# Define dataset root path in Kaggle
dataset_root = "Augmented IQ-OTHNCCD lung cancer dataset"  # Change to your local dataset folder

# List the dataset structure
print("Dataset Structure:")
#!ls -lh "{dataset_root}"


# ## Count number of images in each class(checking data imbalace)

# In[20]:


# Define paths to class folders
class_folders = ["Malignant cases", "Normal cases"]


class_counts = {}
for class_name in class_folders:
    class_path = os.path.join(dataset_root, class_name)
    num_images = len(os.listdir(class_path))  # Count files
    class_counts[class_name] = num_images

# Display results
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

plt.figure(figsize=(8, 6))
plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'red'])
plt.xlabel("Class Type")
plt.ylabel("Number of Images")
plt.title("Lung Cancer Dataset Class Distribution")
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# ## Check File Formats

# In[11]:




# List a few image filenames from each class
for class_name in ["Malignant cases", "Normal cases"]:
    class_path = os.path.join(dataset_root, class_name)
    sample_files = os.listdir(class_path)[:5]  # Get first 5 files
    print(f"\n{class_name} sample images:")
    print(sample_files)


# ## Display Sample Images

# In[12]:






# Function to display images
def show_sample_images(class_name, num_samples=5):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg'))][:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{class_name} - {img_file}")
        axes[i].axis("off")

    plt.show()

# Display samples for each class
for category in ["Malignant cases", "Normal cases"]:
    show_sample_images(category)


# ## Function to check image properties

# In[13]:


import cv2
import numpy as np


def check_image_properties(class_name, num_samples=5):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]

    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)  # Load the image
        height, width, channels = img.shape  # Get shape
        
        print(f"{img_file}:")
        print(f" - Dimensions: {width}x{height}")
        print(f" - Channels: {channels} (1=Grayscale, 3=RGB)")
        print(f" - Min Pixel Value: {np.min(img)}, Max Pixel Value: {np.max(img)}\n")

# Check properties for each class
for category in ["Malignant cases", "Normal cases"]:
    print(f"\nChecking images in {category}...")
    check_image_properties(category)


# ## preprocess images for better fit in resnet ( images should be RGB , and resized to 224*224)

# In[14]:


import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import shuffle

# Standard image size
IMG_SIZE = 224  

# Function to preprocess images (keeping RGB format)
def preprocess_images(class_name):
    class_path = os.path.join(dataset_root, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    processed_images = []
    labels = []
    
    # Label mapping
    label_dict = {"Malignant cases": 1, "Normal cases": 0} #0 no cancer ; 1 cancer

    for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
        img_path = os.path.join(class_path, img_file)

        # Load image in RGB (No Grayscale)
        img = cv2.imread(img_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV default) to RGB

        # Resize to standard size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values (0 to 1)
        img = img / 255.0

        # Append to list
        processed_images.append(img)
        labels.append(label_dict[class_name])

    return np.array(processed_images), np.array(labels)

# Process all images
# X_benign, y_benign = preprocess_images("Benign cases")
X_malignant, y_malignant = preprocess_images("Malignant cases")
X_normal, y_normal = preprocess_images("Normal cases")

# Combine all classes
X = np.concatenate([X_malignant, X_normal], axis=0)
y = np.concatenate([y_malignant, y_normal], axis=0)

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

# Print final dataset details
print(f"Final dataset shape: {X.shape}, Labels shape: {y.shape}")


# ## Training the model
# ### 5 epoches for fully connected while other layers are freezed, 5 epoches for fine tuning few last layers 

# In[15]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Split Data into Train, Validation, and Test Sets**
y_tensor = torch.tensor(y, dtype=torch.long)  # Convert labels to PyTorch tensor

# Split dataset (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_tensor, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

#To handle class imbalance
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

y_train_np = y_train.cpu().numpy()

class_weights = compute_class_weight(
    class_weight="balanced",
    classes= np.unique(y_train_np),
    y=y_train_np
)

class_weights= torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights:", class_weights)

#  **Define Custom PyTorch Dataset**
class LungCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert to Tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)  # Change (224,224,3) → (3,224,224) for PyTorch

        if self.transform:
            img = self.transform(img)

        return img, label

# Define Transformations
transform = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize between -1 and 1
])

# Create Dataset and DataLoaders
train_dataset = LungCancerDataset(X_train, y_train, transform=transform)
val_dataset = LungCancerDataset(X_val, y_val, transform=transform)
test_dataset = LungCancerDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#  **Load Pretrained ResNet-50 and Modify Last Layer**
resnet50 = models.resnet50(pretrained=True)

# Freeze all layers (train only the last FC layer first)
for param in resnet50.parameters():
    param.requires_grad = False

# Replace the last FC layer
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 2)  # two output classes: normal and malignant

# Move model to GPU
resnet50 = resnet50.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.0001)

#  **Train Only the Last Fully Connected Layer First**
num_epochs = 5
for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

#  **Unfreeze Last Few Layers and Fine-Tune**
for param in list(resnet50.layer4.parameters()):  # Unfreeze last residual block
    param.requires_grad = True

optimizer = optim.Adam(resnet50.parameters(), lr=0.00001)  # Fine-tuning learning rate

num_finetune_epochs = 5
for epoch in range(num_finetune_epochs):
    resnet50.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet50(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_acc = 100 * correct / total
    print(f"Fine-Tune Epoch {epoch+1}/{num_finetune_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")


# ## **Evaluate on Validation Set**

# In[16]:



resnet50.eval()  # Set model to evaluation mode
correct = 0
total = 0
val_loss = 0.0
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Define loss function for validation

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute final validation accuracy and loss
val_acc = 100 * correct / total
val_loss /= len(val_loader)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.2f}%")


# ## **Save Model**

# In[17]:



torch.save(resnet50.state_dict(), "resnet50_lung_cancer_binary_detector_training.pth")
print("Model saved successfully!")

# ##  **Evaluate on Test Set**

# In[18]:
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


resnet50.eval()  # Set model to evaluation mode
correct = 0
total = 0
test_loss = 0.0
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Define loss function for test evaluation

y_true = []
y_pred = []

with torch.no_grad():  # No gradient calculation during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50(images)
        
        # Compute test loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute final test accuracy and loss
test_acc = 100 * correct / total
test_loss /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")


# Test metrics are printed above from the current run.
# Do not substitute accuracy values copied from another experiment.

# In[ ]:

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Cancer","Cancer"],
            yticklabels=["Non-Cancer", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Lung Cancer Detection")
plt.show()

print(f"Evaluated images: {len(y_true)}")
print("Actual class counts (0=non-cancer, 1=cancer):", np.bincount(np.asarray(y_true), minlength=2))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Non-Cancer", "Cancer"]))

