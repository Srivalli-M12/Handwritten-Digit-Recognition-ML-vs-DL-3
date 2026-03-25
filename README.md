# Handwritten-Digit-Recognition-ML-vs-DL-3
This project explores and compares the performance of multiple traditional Machine Learning algorithms against a Deep Learning Convolutional Neural Network (CNN) in classifying handwritten digits from the classic MNIST dataset.

## 📊 Dataset
The project utilizes the **MNIST Database**, which contains 60,000 training images and 10,000 testing images of handwritten digits (0-9), formatted as 28x28 pixel grayscale grids.

## 🚀 Models & Performance
Four different models were trained and evaluated to find the most accurate classifier. The CNN outperformed the traditional machine learning models, achieving a highly accurate peak score.

| Model | Type | Test Accuracy |
| :--- | :--- | :--- |
| **K-Nearest Neighbors (KNN)** | Machine Learning | 96.87% |
| **Random Forest Classifier** | Machine Learning | 97.02% |
| **Support Vector Machine (SVM)** | Machine Learning | 97.85% |
| **Convolutional Neural Network (CNN)** | Deep Learning | **99.26%** |

## 🛠️ Tech Stack
* **Language:** Python 3
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-Learn
* **Data Processing & Visualization:** NumPy, OpenCV, Matplotlib

## 📂 Project Structure
* Each model is isolated in its own script/notebook for easy testing.
* Traditional models utilize a custom `mnist_loader` to parse the raw IDX binary files.
* The CNN utilizes TensorFlow's modernized `fetch_openml` capabilities and runs optimally on a GPU environment (like Google Colab's T4).

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install numpy opencv-python scikit-learn tensorflow matplotlib`
3. For traditional ML models, ensure the unzipped MNIST dataset is placed in the `MNIST_Dataset_Loader/dataset/` directory.
4. Run the individual model scripts to view the training process, accuracy scores, and confusion matrix visualizations.
