import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define dataset paths
real_path = "ml\\real_and_fake_face\\training_real"
fake_path = "ml\\real_and_fake_face\\training_fake"

# Force CUDA to be used if available
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("WARNING: GPU not detected. Running on CPU will be very slow.")
print(f"Using device: {device}")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [(os.path.join(real_dir, img), 1) for img in os.listdir(real_dir)]
        self.fake_images = [(os.path.join(fake_dir, img), 0) for img in os.listdir(fake_dir)]
        self.all_images = self.real_images + self.fake_images
        random.shuffle(self.all_images)
        self.transform = transform
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a fallback to prevent training crash
            fallback_img = torch.zeros((3, 224, 224))
            return fallback_img, label

print("Loading dataset...")
full_dataset = DeepfakeDataset(real_path, fake_path, transform=train_transform)

# Split into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply different transforms to validation dataset
val_dataset.dataset.transform = val_transform

num_fake = len(os.listdir(fake_path))
num_real = len(os.listdir(real_path))
total = num_fake + num_real
class_weights = torch.tensor([total / num_fake, total / num_real], dtype=torch.float32).to(device)

batch_size = 32
num_workers = 4 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

class DeepfakeDetector(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DeepfakeDetector, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Number of features in the last layer
        in_features = self.resnet.fc.in_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  
            nn.Linear(128, 2) 
        )
        
        # Replace the original classifier
        self.resnet.fc = self.classifier
        
    def forward(self, x):
        return self.resnet(x)

# Initialize the model
print("Initializing model...")
model = DeepfakeDetector(dropout_rate=0.5).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    best_val_loss = float('inf')
    best_model_state = None
    
    # For tracking metrics
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [],'val_acc': []}
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! Validation Loss: {best_val_loss:.4f}")
        
        # Record metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print("-" * 60)
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return model, history

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to evaluate model on validation set
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "real" class
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Fake', 'Real'],
               yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    print(report)
    return cm, report, roc_auc

# Function to visualize model predictions
def visualize_predictions(model, data_loader, num_samples=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_samples//5 + 1, 5, images_so_far)
                ax.axis('off')
                
                # Convert tensor to image
                img = inputs[j].cpu().numpy().transpose((1, 2, 0))
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                ax.set_title(f'Pred: {["Fake", "Real"][preds[j]]}\nTrue: {["Fake", "Real"][labels[j]]}\nProb: {probs[j][preds[j]]:.2f}',
                           color=('red' if preds[j] != labels[j] else 'green'))
                
                if images_so_far == num_samples:
                    plt.tight_layout()
                    plt.savefig('predictions_visualization.png')
                    plt.show()
                    return
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()

# Function to save and load models
def save_model(model, filename='deepfake_detector_resnet50.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, filename)
    print(f"Model saved to {filename}")

def load_model(model, filename='deepfake_detector_resnet50.pth'):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filename}")
    return model

# Function to make predictions on new images
def predict(model, image_path):
    model.eval()
    transform = val_transform
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    pred_class = "Real" if preds.item() == 1 else "Fake"
    confidence = probs[0][preds.item()].item() * 100
    
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.2f}%)")
    plt.axis('off')
    plt.show()
    
    return pred_class, confidence

# Function to apply Grad-CAM for model interpretability
def apply_grad_cam(model, image_path):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        # Define the target layer (usually the last convolutional layer for ResNet50)
        target_layer = model.resnet.layer4[-1]
        
        # Create a GradCAM object
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
        
        # Preprocess the image
        transform = val_transform
        rgb_img = Image.open(image_path).convert('RGB')
        input_tensor = transform(rgb_img).unsqueeze(0).to(device)
        
        # Get the CAM
        grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert PIL image to numpy array and normalize
        rgb_img_array = np.array(rgb_img) / 255.0
        
        # Overlay heatmap on original image
        visualization = show_cam_on_image(rgb_img_array, grayscale_cam, use_rgb=True)
        
        # Display the result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img_array)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title('Grad-CAM Visualization')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('grad_cam.png')
        plt.show()
    except ImportError:
        print("pytorch-grad-cam package not found. Please install with: pip install pytorch-grad-cam")

# Main execution block
# if __name__ == "__main__":
#     try:
        
#         # Train the model
#         print("Starting model training...")
#         model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)
        
#         # Plot training history
#         plot_training_history(history)
        
#         # Evaluate the model
#         print("Evaluating model on validation set...")
#         cm, report, roc_auc = evaluate_model(model, val_loader)
        
#         # Visualize some predictions
#         print("Visualizing predictions...")
#         visualize_predictions(model, val_loader)
        
        
#         # Save the model
#         save_model(model)
        
#         print("Training and evaluation complete!")
#         # Optional: Test on a single image
#         test_image_path = "ml\\krishna.jpg"
#         predict(model, test_image_path)
        
#         # Optional: Apply Grad-CAM for interpretability
#         # apply_grad_cam(model, test_image_path)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
if __name__ == "__main__":
    try:
        model_path = "ml\\deepfake_detector_resnet50.pth"

        if os.path.exists(model_path):
            print(f"[INFO] Found existing model at '{model_path}'. Loading...")
            model = load_model(model, filename=model_path)
        else:
            print("[INFO] No saved model found. Starting training from scratch...")

            # Train the model
            print("[INFO] Starting model training...")
            model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

            # Plot training history
            plot_training_history(history)

            # Save the model after training
            save_model(model, filename=model_path)

        # Evaluate the model
        print("[INFO] Evaluating model on validation set...")
        cm, report, roc_auc = evaluate_model(model, val_loader)

        # Visualize some predictions
        print("[INFO] Visualizing predictions...")
        visualize_predictions(model, val_loader)

        # Optional: Test on a single image
        test_image_path = "ml\\krishna.jpg"
        predict(model, test_image_path)

        # Optional: Apply Grad-CAM for interpretability
        # apply_grad_cam(model, test_image_path)

        print("[INFO] Training and evaluation complete!")

    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
