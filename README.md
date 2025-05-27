🧠 Brain Tumor Detection from MRI Images using CNNs

This project leverages Convolutional Neural Networks (CNNs) with TensorFlow to detect brain tumors from MRI images. It covers data preprocessing, model training, evaluation, and visualization. Perfect for deep learning and medical imaging enthusiasts! 🩺💡
📂 Dataset Structure

Make sure your dataset is organized like this:

brain_mri/brain_tumor_dataset/
├── yes/    # ➕ Tumor present
└── no/     # ➖ No tumor

🖼️ Images are automatically resized to 128x128 and normalized for training.
🛠️ Requirements

Install dependencies using pip:

pip install tensorflow numpy matplotlib seaborn scikit-learn

📦 Dependencies used:

    TensorFlow 📊

    NumPy 🔢

    Matplotlib 📈

    Seaborn 🎨

    scikit-learn 📚

🚀 How to Run

    📥 Clone the repo or copy the script.

    📁 Ensure your dataset is inside: brain_mri/brain_tumor_dataset/

    ▶️ Run the Python script or Jupyter notebook.

🧱 Project Breakdown
1️⃣ Imports

All essential libraries for deep learning and visualization.
2️⃣ Data Preparation

    ✅ Rescaling and normalization

    🔀 Train-validation split (80-20)

    📐 Resize to 128x128 RGB

3️⃣ CNN Architecture

🧠 Built using Sequential model:

    🧩 Conv2D + ReLU

    🚿 MaxPooling

    🪜 Flatten + Dense

    🎯 Softmax output

4️⃣ Training

🛠 Optimizer: Adam
📉 Loss Function: categorical_crossentropy
📆 Epochs: 10
5️⃣ Accuracy & Loss Plots

📈 Visual graphs to monitor training and validation performance.
6️⃣ Predictions

🔍 View model predictions alongside true labels (8 samples).
7️⃣ Evaluation

📊 Confusion Matrix
🧾 Classification Report (Precision, Recall, F1-score)
🧪 Sample Output

✅ Model Accuracy & Loss Plot
🖼 Predicted vs. Actual Image Labels
📉 Confusion Matrix
🧾 Classification Report
⚙️ Customization Tips

🛠️ Modify these parameters as needed:

    data_dir → Change path to your dataset

    target_size, batch_size → Adjust for speed or accuracy

    epochs → Try higher values for improved accuracy

📌 Notes

    The script detects the number of classes automatically.

    Works with RGB images only. Convert grayscale if necessary.

📃 License

🧑‍🎓 For educational and research use only. Please cite the dataset's original source if used in academic publications.
