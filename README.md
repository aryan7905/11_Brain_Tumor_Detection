🧠 <span style="color:#4B8BBE;">Brain Tumor Detection from MRI Images using CNNs</span>
<span style="color:#43A047;">Project Overview</span>

This project implements a Convolutional Neural Network (CNN) to classify brain MRI images for detecting brain tumors. Using a publicly available dataset from Kaggle, the model learns to distinguish between different types of brain tumors and non-tumor images. The workflow includes data preprocessing, model training, evaluation, and visualization of results.
<span style="color:#1976D2;">Dataset</span>

The dataset used is the Brain MRI Images for Brain Tumor Detection from Kaggle. It contains MRI images categorized into multiple classes such as Glioma tumor, Meningioma tumor, Pituitary tumor, and No tumor.
✨ Features

    Data loading and augmentation using Keras ImageDataGenerator 📊

    Custom CNN model architecture for multi-class classification 🏗️

    Training and validation with accuracy and loss visualization 📈

    Confusion matrix heatmap for detailed performance evaluation 🔥

    Visualization of sample predictions with true and predicted labels 👁️‍🗨️

🛠️ Installation and Setup

    Clone the repository:

bash
git clone https://github.com/aryan7905/brain-tumor-detection.git
cd brain-tumor-detection

Install required packages:

    bash
    pip install -r requirements.txt

    Main dependencies: TensorFlow, Keras, Matplotlib, Seaborn, scikit-learn.

    Download the dataset from Kaggle and upload it to your working directory or Google Colab environment.

🚀 Usage

    Upload the dataset folder (unzipped) to your environment.

    Run the Jupyter notebook or Python script to train the CNN model:

        The notebook includes cells for data preprocessing, model building, training, and evaluation.

        Visualizations of training progress and predictions are generated automatically.

    Evaluate the model using the confusion matrix and classification report.

📸 Example Output

    Training Accuracy and Loss Curves![{AC5F5EA3-61C9-46B7-A57E-82170DE10D31}](https://github.com/user-attachments/assets/d88a3531-0acb-4727-8fc1-6319006a4124)


    Confusion Matrix Heatmap![{91DD8556-57DF-4E49-B8A8-E704F1C98841}](https://github.com/user-attachments/assets/5ea4bf5d-d31b-4430-a819-81c4e622efe2)


    Sample Predictions with True vs Predicted Labels![{DAAC0979-D097-4F7D-B355-E5FD9CF59E69}](https://github.com/user-attachments/assets/449b461e-dc5f-4d67-a3e8-aae99535ed83)


<div align="center"> <!-- Place your confusion matrix screenshot here --> <img src="path_to_confusion_matrix_image.png" alt="Confusion Matrix" width="400"/> <!-- Place your model architecture screenshot here --> <img src="path_to_model_architecture_image.png" alt="Model Architecture" width="400"/> </div>
🔮 Future Work

    Implement transfer learning with pretrained models (e.g., EfficientNet, ResNet) 💡

    Add Grad-CAM visualization for model interpretability 🔍

    Experiment with hyperparameter tuning for improved accuracy ⚙️

    Deploy the model as a web app for real-time tumor detection 🌐

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Dataset provided by Navoneel Chakrabarty on Kaggle

    TensorFlow and Keras libraries for deep learning tools
