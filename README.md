# Feature-Based Bird Classification Using the CUB-200-2011 Dataset

## Overview
This project focuses on classifying bird species using machine learning models while providing explainability into model decisions. The dataset used is **The Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset, which consists of images of 200 bird species with detailed annotations. The primary goal is to classify birds based on their visual features and associated attributes. We explore multiple machine learning classifiers, compare their performance, and investigate feature importance.

## Dataset Description
The dataset contains the following files:

- **images.txt** - Contains Image ID and Image file name.
- **image_class_labels.txt** - Contains Image ID and Class ID.
- **classes.txt** - Contains Class ID and Class name.
- **attributes.txt** - Contains Attribute ID and Attribute name.
- **image_attribute_labels.txt** - Contains Image ID, Attribute ID, and Is Present.
- **train_test_split.txt** - Contains Image ID and Is Training Image.

## Data Preprocessing

1. **Load the Data**: The dataset is loaded and preprocessed using Python. Each file is parsed to create appropriate mappings between image IDs, class labels, and attributes.
   
2. **Prepare the Data**: Images are resized and normalized for machine learning models. We handle missing values and apply feature engineering to extract relevant attributes that aid in classification.

3. **Train the Model**: Several classifiers, including Support Vector Machines (SVM), Decision Trees, and Random Forests, are trained on the data using a variety of feature sets and hyperparameters.

4. **Make Predictions**: Once the model is trained, predictions are made on the test set, and results are evaluated using various performance metrics.

5. **Evaluate the Model**: Performance metrics such as accuracy, confusion matrix, and F1-score are used to assess the effectiveness of the classifiers.

---

## Classification Methods

### 1. Support Vector Machine (SVM)

SVM is used to classify bird species by finding the optimal hyperplane to separate different classes. Various kernels (Linear, RBF) were tested to determine the best-performing model.

#### Hyperparameters:
- **Kernel = "Linear"** | Accuracy = **43.87%**
- **Kernel = "RBF" (Radial Basis Function)** | Accuracy = **45.63%**
- **Grid Search CV Optimization** | Accuracy = **46.67%**
![](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/SVM%20Images/SVM%20Accuracy.png)

#### Performance Metrics:
- **Confusion Matrix**  
  ![Confusion Matrix](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/SVM%20Images/Confusion%20Matrix.png)
- **F1-Score**  
  ![F1-Score](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/SVM%20Images/F1ScoreForALL.png)

#### Insights:
- The **RBF kernel** tends to perform better than the linear kernel, likely due to its ability to handle non-linear boundaries in the data.
- **Grid Search CV** optimization fine-tunes the hyperparameters and improves accuracy.

### 2. Decision Tree

The decision tree classifier splits data into subgroups based on feature importance and splits at various decision thresholds to classify bird species.

#### Performance:
- **Decision Tree Accuracy**: **25.01%**
- **Random Forest Accuracy**: **43.61%**
- **Random Forest with Grid Search CV**: **44.56%**
![](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/DT%20Images/DT%20Accuracy.png)
#### Performance Metrics:
- **Confusion Matrix**  
  ![Confusion Matrix](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/DT%20Images/Confusion%20Matrix.png)
- **F1-Score**  
  ![F1-Score](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/DT%20Images/F1%20Score%20for%20all%20Classes.png)

#### Insights:
- Decision trees showed relatively low accuracy, possibly due to overfitting or lack of sufficient depth in the tree.
- Random forests, being an ensemble of decision trees, showed an improvement in performance.

---

## Accuracy vs. Number of Attributes

Using SVM coefficients, we visualized the importance of various attributes in classification, such as **wing shape**, **beak type**, and **color patterns**. As the number of attributes increases, the accuracy of the model improves, which suggests that more detailed features provide useful information for distinguishing between bird species.

![Accuracy vs. Attributes](https://github.com/Zeeshan6948/xAI_final_proj/blob/main/DT%20Images/Accuracy250_260.png)

---

## Results and Discussion

- The SVM with RBF kernel consistently outperformed other models, achieving an accuracy of **45.63%**. 
- The decision tree, while less accurate, provided valuable insights into the importance of features used for classification. However, it struggled with overfitting.
- Random forests performed better than individual decision trees, but the improvement was marginal.

Overall, the accuracy could still be improved by exploring deeper models such as neural networks and integrating more complex features or attribute sets.

---

## Future Work

- **Deep Learning Models**: Exploring convolutional neural networks (CNNs) and transfer learning for image-based feature extraction.
- **Additional Data**: Incorporating more bird images or data from other sources could enhance model performance.
- **Explainability**: Further development of tools like LIME or SHAP could help explain model predictions in more detail, providing better insights into the decision-making process.

---

## Installation & Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn
  ```

### Steps to run the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/Zeeshan6948/xAI_final_proj.git
   ```

2. Navigate to the project directory:
   ```bash
   cd xAI_final_proj
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

---

## References

- CUB-200-2011 dataset: [http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- Support Vector Machines: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
- Decision Trees and Random Forests: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

---

**Authors:** Zeeshan Ahmed  
**Date:** January 30, 2025
