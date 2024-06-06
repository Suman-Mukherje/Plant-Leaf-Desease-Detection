import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(csv_file):
    data = pd.read_csv(csv_file)
    epochs = data['Epoch']
    train_loss = data['Train Loss']
    val_loss = data['Val Loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(csv_file):
    data = pd.read_csv(csv_file)
    epochs = data['Epoch']
    train_acc = data['Train Accuracy']
    val_acc = data['Val Accuracy']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    csv_file = 'cnn_metrics.csv'
    plot_loss(csv_file)
    plot_accuracy(csv_file)
