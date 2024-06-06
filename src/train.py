import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, datasets, transforms
from model import CNNModel
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Define data transformations (you can customize these based on your needs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the root folder where your data is stored
data_root = '/content/drive/MyDrive/plantleafsmall'

# Create the ImageFolder dataset
dataset = datasets.ImageFolder(root=data_root, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size= 64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print the number of samples in each set
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(torch.device('cuda'))
        target = target.to(torch.device('cuda'))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    train_accuracy = 100 * correct / total

    return running_loss / len(train_loader.dataset), train_accuracy

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    return val_loss / len(val_loader.dataset), val_accuracy

# Testing loop
def test(model, test_loader):
    model.eval()
    model.to('cuda')  # Move the model to GPU
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    return test_accuracy

# Open a CSV file to store the metrics
with open('cnn_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Test Accuracy'])

    # Training the model
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        test_acc = test(model, test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%, Test Acc: {test_acc:.4f}%")

        # Write the metrics to the CSV file
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, test_acc])

print("Training metrics saved to cnn_metrics.csv")

# Save the trained model
torch.save(model.state_dict(), 'cnn_leaf.pth')
