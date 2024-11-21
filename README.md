# AI Skin Cancer Detection

This project uses a fine-tuned ResNet50 model to classify different types of skin lesions, leveraging the HAM10000 Skin Cancer dataset. The goal is to assist in early detection of skin cancer and improve diagnostic accuracy through AI-driven solutions.

## Dataset

**HAM10000**: A collection of 10,000 dermatoscopic images categorized into seven different types of skin lesions. The dataset includes:

- Actinic keratoses (AKIEC)
- Basal cell carcinoma (BCC)
- Benign keratosis-like lesions (BKL)
- Dermatofibroma (DF)
- Melanoma (MEL)
- Melanocytic nevi (NV)
- Vascular lesions (VASC)

For more details, visit the [HAM10000 dataset repository](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

## Model

- **Architecture**: ResNet50
- **Fine-Tuning**: The model was pre-trained on ImageNet and fine-tuned on the HAM10000 dataset.
- **Objective**: Multi-class classification of skin lesion types.

### Key Features:
- **Transfer Learning**: Utilizes the robust feature extraction capabilities of ResNet50.
- **Data Augmentation**: Includes rotation, flipping, zoom, and other augmentations to enhance model robustness.
- **Regularization**: Employs dropout and weight decay to prevent overfitting.
