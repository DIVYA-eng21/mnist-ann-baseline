# MNIST Digit Classification using Neural Network (Keras)

## 📌 Project Overview

This project implements a **fully connected neural network (ANN)** to classify handwritten digits (0–9) from the **MNIST dataset** using **TensorFlow / Keras**.
The model takes grayscale images of size **28×28** and predicts the digit present in the image.

This is a **baseline deep learning model**, useful for understanding:

* Input shapes
* Flattening image data
* Dense layers
* Dropout for regularization
* Softmax-based multiclass classification

---

## 🧠 Model Architecture (What is happening internally)

```

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
```

### Layer-by-layer explanation:

1. **Flatten (28×28 → 784)**

   * Converts each image into a 1D vector of 784 pixels
   * Required because Dense layers accept 1D input

2. **Dense (128 units, ReLU)**

   * Learns high-level pixel patterns
   * ReLU introduces non-linearity

3. **Dropout (0.2)**

   * Randomly disables 20% of neurons during training
   * Prevents overfitting by reducing memorization

4. **Dense (32 units, ReLU)**

   * Further feature compression and abstraction

5. **Dropout (0.2)**

   * Additional regularization

6. **Dense (10 units, Softmax)**

   * Outputs probabilities for digits 0–9
   * Softmax ensures probabilities sum to 1

---

## 📂 Dataset

* **MNIST Handwritten Digits**
* Training set: 60,000 images
* Test set: 10,000 images
* Image size: 28×28 (grayscale)

---

## ⚙️ Model Compilation

Typical compilation setup:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

* **Adam**: Adaptive learning rate optimizer
* **Sparse categorical crossentropy**: Used because labels are integers (0–9)

---

## 🚀 Training

```python
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32 #default .(not written in code.)
)
```

* Validation data helps detect overfitting
* Dropout improves generalization

---

## 📈 Results & Observations

* Training loss decreases rapidly
* Validation loss may plateau or slightly increase
* Indicates **mild overfitting**, which is expected for ANN on image data
* Dropout reduces but does not eliminate overfitting

---


## 🔮 Future Improvements

* Replace Dense layers with **Conv2D + MaxPooling**
* Add **BatchNormalization**
* Tune dropout rate
* Use **data augmentation**

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab (GPU: T4)

---

## ✅ Conclusion

This project demonstrates a clean and correct implementation of a neural network for digit classification.

---

**Author:** Divya
**Domain:** Deep Learning / Neural Networks
**Dataset:** MNIST
