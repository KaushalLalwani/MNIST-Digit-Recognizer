# **MNIST Handwritten Digit Classifier**

This project implements a **Convolutional Neural Network (CNN)** for handwritten digit recognition using the **MNIST dataset**. The model is built using **TensorFlow/Keras**, and an optional **Streamlit web app** is included for real-time digit predictions.

---

## 🚀 **Project Overview**

✅ **Train a CNN model** to classify handwritten digits (0-9).  
✅ **Use the MNIST dataset** for training and evaluation.  
✅ **Preprocess user-input images** for prediction.  
✅ **Deploy an interactive web app** using Streamlit (optional).  


## 🏗 **Installation & Setup**

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/yourusername/MNIST-Digit-Recognizer.git
cd MNIST-Digit-Recognizer
```

### 2️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 🏋️ **Train the Model**

Run the following command to train the CNN model on the MNIST dataset:
```sh
python src/train_model.py
```
🔹 This will save the trained model as `mnist_digit_recognizer.h5` inside the `model/` directory.

---

## 🎯 **Test the Model on User Images**

To predict a digit from an uploaded image:
```sh
python src/predict_digit.py --image dataset/sample_images/digit.png
```
🔹 This script loads the trained model and processes the input image for prediction.

---

## 🌐 **Run the Web App (Optional)**

For an interactive **Streamlit** web interface:
```sh
streamlit run app.py
```
🔹 Upload a handwritten digit image, and the app will display the prediction.

---

## 📊 **Results & Visualizations**

Training accuracy and loss curves:
```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## 🔥 **Future Improvements**

🚀 Implement **Transfer Learning** with a pre-trained model like MobileNet.  
🚀 Add support for **real-time digit recognition** using a webcam.  
🚀 Deploy the model as a **Flask/FastAPI** API for cloud-based predictions.  

---

## 🤝 **Contributing**
Feel free to fork this repository, improve the model, and submit pull requests!

---

## 📝 **License**
This project is licensed under the MIT License.

---

📌 **Author:** [Kaushal Lalwani] 
📌 **GitHub Repository:** [MNIST-Digit-Recognizer](https://github.com/KaushalLalwani/MNIST-Digit-Recognizer)

