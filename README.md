# **Image Classification via Pre-trained ResNet Models**

This repository contains a practical, hands-on deep learning workspace built with **PyTorch** to perform automated image recognition. The project loads the popular `CIFAR-10` dataset, prepares raw image files using advanced data modification techniques, fine-tunes a pre-trained professional vision system (**ResNet**), and saves the trained model to classify completely unseen images on demand.

## 📂 Project Structure
*   `tony_u4_l6_pytorch1.ipynb` - The full Jupyter Notebook containing data pipeline engineering, interactive model selection, GPU training loops, and live image classification metrics.
* **[Training AI Pre-trained Models](https://github.com/anthonymroso-star/ai_transfer_learning/blob/main/notebook/tony_u4_l6_pytorch1.ipynb)** - AI Transfer Learning, using the Pre-trainied ResNet model to transfer its visual knowlege into a customised class image identifier.
##  Core Workflow Steps

### 1. Data Augmentation & Pipelines
*   **Image Transformations:** Utilises **Torchvision** to apply live data adjustments like random cropping, horizontal/vertical mirroring, and rotation to train a more adaptable system.
*   **Data Splitting:** Segregates the image pool into three independent groups—**Training (80%)**, **Validation (10%)**, and **Testing (10%)**.
*   **Batch Delivery:** Creates systematic data loaders that group data into structured batches of 32 images and shuffles training pathways to prevent memorization.

### 2. Transfer Learning Architecture & Adaptation
*   **Knowledge Transfer:** Implements a **Transfer Learning** strategy by loading a pre-trained vision model (**ResNet18** or **ResNet34**) that already understands complex visual features (like shapes, edges, and textures) learned from millions of general internet images.
*   **Output Rewiring:** Cuts off the model's original final classification layer and replaces it with a new layer tailored specifically to your 10 target categories (including airplanes, automobiles, birds, and cats).

*   **Output Rewiring:** Automatically swaps out the original final layer of the network to match the specific 10-class dataset architecture (including categories like airplanes, automobiles, birds, and cats).

### 3. GPU-Accelerated Training Loop
*   **Hardware Maximisation:** Automatically checks for a dedicated graphics processor (**GPU CUDA**) to speed up training tasks, defaulting safely to standard CPU processing when needed.
*   **Optimisation Setup:** Deploys a Cross-Entropy loss tracker paired with Stochastic Gradient Descent (SGD) to measure classification mistakes and update parameters over 10 training rounds (epochs).
*   **Loss Tracking:** Demonstrates steady system improvement as the measured error drops consistently from an initial score of 1.15 down to 0.60 by the final round.

### 4. Saving Checkpoints & Live Prediction
*   **Model Saving:** Exports the finalised, trained internal weights into a portable deployment file named `model.pth`.
*   **Live Image Inference:** Downloads a random sample picture (such as an A380 airliner) from the web, formats it to match original dimensions, and feeds it into the quieted evaluation engine.
*   **Classification Results:** Successfully categorises new test images with high accuracy, automatically turning mathematical output matrix scores back into clear index values (like identifying the sample airplane as Category 0).

