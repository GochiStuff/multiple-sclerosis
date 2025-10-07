# MRI Analysis with Deep Learning

This repository contains three Jupyter notebooks demonstrating different machine learning techniques on Multiple Sclerosis (MS) MRI datasets.

+

One complete application to run the resnet model offline. for prediction.

---
### To run the model offline . 

# MRI AI Project - Quick Start

Run this project locally using Docker.

### Prerequisites
* [Docker](https://www.docker.com/get-started) must be installed and running.
else create a virtual python environment before installing the dependencies. as the packages can take 10 - 16 Gbs.
### Instructions

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone [https://github.com/GochiStuff/multiple-sclerosis.git](https://github.com/GochiStuff/multiple-sclerosis.git)
    cd multiple-sclerosis
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t mri-ai:v1 .
    ```

3.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 mri-ai:v1
    ```

4.  **Access the app:**
    Open your web browser and go to `http://localhost:5000`.
---

### 1. `01-CBIR-InceptionV3-FAISS.ipynb`

* **Task**: Content-Based Image Retrieval (CBIR).
* **Description**: This notebook builds a system to find visually similar MRI slices. It uses a pre-trained **InceptionV3** model with TensorFlow/Keras to extract feature embeddings from MRI slices. **FAISS** is then used to index these features and perform an efficient similarity search based on Euclidean distance.
* **Outcome**: Given a query image, it retrieves and displays the most similar-looking slices from the dataset.

---

### 2. `02-Classification-ResNet18-PyTorch.ipynb` -> HIGHEST ACCURACY 

* **Task**: Binary Image Classification.
* **Description**: This notebook trains a classifier to distinguish between MRI slices of MS patients and healthy controls. It uses **transfer learning** by fine-tuning a pre-trained **ResNet-18** model with **PyTorch**. The script handles class imbalance and includes data augmentation.
* **Outcome**: A trained model that achieves **~98% accuracy** on the test set, with a full evaluation including a classification report and confusion matrix.

---

### 3. `03-Lesion-Detection-InceptionV3-TensorFlow.ipynb` -> BEST BUT LOW ACCURACY

* **Task**: Lesion Detection (Slice-level and Patient-level classification).
* **Description**: This notebook trains a binary classifier to detect the presence or absence of MS lesions on individual MRI slices. It uses a fine-tuned **InceptionV3** model with TensorFlow/Keras and evaluates performance at both the slice and patient level by aggregating predictions.
* **Outcome**: A diagnostic model that predicts the probability of a lesion on each slice and provides an overall patient diagnosis, visualized with ground truth overlays.

---
