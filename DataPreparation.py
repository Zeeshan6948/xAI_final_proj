import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load data
dataset_path = R'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011'

# Load attributes (continuous values for each class)
attributes = np.loadtxt(f'{dataset_path}/attributes/class_attribute_labels_continuous.txt')

# Load class labels (1-based indexing)
class_labels = np.loadtxt(f'{dataset_path}/image_class_labels.txt', dtype=int)[:, 1]

# Map species-level attributes to image-level attributes
# Each image gets the attribute vector of its species
image_attributes = attributes[class_labels - 1]  # Subtract 1 for zero-based indexing

# Now `image_attributes` has the same number of rows as `class_labels`
print(f"Image attributes shape: {image_attributes.shape}")  # Should match (11788, 312)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(image_attributes, class_labels, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
# Normalize attributes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train - 1, dtype=torch.long)  # Subtract 1 for zero-indexing
y_test_tensor = torch.tensor(y_test - 1, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class AttributeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AttributeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = X_train.shape[1]
num_classes = 200
model = AttributeClassifier(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {correct / total * 100:.2f}%")


from torchvision import models, transforms
from PIL import Image

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = nn.Identity()  # Remove the classifier for feature extraction
vgg16.eval()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from an image
def extract_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = vgg16(image_tensor)
    return features.flatten()

# Example usage
image_path = r"C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\images\197.Marsh_Wren\Marsh_Wren_0099_188579.jpg";
image_features = extract_image_features(image_path)

# Define the model
class ImageToAttribute(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImageToAttribute, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

image_to_attr_model = ImageToAttribute(input_size=512 * 7 * 7, output_size=X_train.shape[1])  # Adjust input size for VGG16

# Predict attributes from the new image features
predicted_attributes = image_to_attr_model(image_features.unsqueeze(0))
# Load class names from `classes.txt`
class_names = {}
with open(f'{dataset_path}/classes.txt', 'r') as f:
    for line in f:
        index, name = line.strip().split(' ', 1)  # Split the line into class index and name
        class_names[int(index)] = name

# Predict species
with torch.no_grad():
    species_prediction = model(predicted_attributes)
    predicted_species_index = torch.argmax(species_prediction).item() + 1  # Add 1 for 1-based indexing

# Map class index to name
predicted_species_name = class_names[predicted_species_index]
print(f"Predicted Bird Species: {predicted_species_name}")
