# Plant Leaf Disease Prediction

This project uses a Convolutional Neural Network (CNN) to predict diseases in plant leaves from images. The model is built using PyTorch and is trained on a dataset of plant leaf images.

## Dataset

The dataset used for this project can be found at [Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network](https://data.mendeley.com/datasets/tywbtsjrjv/1).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Suman-Mukherje/Plant-Leaf-Desease-Detection
    cd Plant-Leaf-Desease-Detection
    ```

### Training the Model

Run the following command to train the model:
```bash
python src/train.py
```

### Evaluating the Model

Run the following command to evaluate the model:
```bash
python src/evaluate.py
```

### Visualizing Metrics

Run the following command to visualize the metrics of the model:
```bash
python src/visualize.py
```

### Results:
From this model we got Accuracy: 91.1581%, Precision: 0.9135, Recall: 0.9092
