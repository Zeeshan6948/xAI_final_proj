# Feature-Based Bird Classification Using the CUB-200-2011 Dataset

## Overview
This project focuses on classifying bird species using machine learning models while providing explainability into model decisions. The dataset used is **The Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset, which consists of images of 200 bird species with detailed annotations.

## Dataset Description
The dataset contains the following files:

- **images.txt** - Contains Image ID and Image file name.
- **image_class_labels.txt** - Contains Image ID and Class ID.
- **classes.txt** - Contains Class ID and Class name.
- **attributes.txt** - Contains Attribute ID and Attribute name.
- **image_attribute_labels.txt** - Contains Image ID, Attribute ID, and Is Present.
- **train_test_split.txt** - Contains Image ID and Is Training Image.

## Data Preprocessing
1. **Load the Data**
2. **Prepare the Data**
3. **Train the Model**
4. **Make Predictions**
5. **Evaluate the Model**

## Classification Methods
### 1. Support Vector Machine (SVM)
SVM is used to classify bird species by finding the optimal hyperplane to separate different classes.

#### Hyperparameters:
- Kernel = "Linear" | Accuracy = **43.87%**
- Kernel = "RBF" (Radial Basis Function) | Accuracy = **45.63%**
- Grid Search CV Optimization | Accuracy = **46.67%**

#### Performance Metrics:
- **Confusion Matrix** (SVM Images/Confusion Matrix.png)
- **F1-Score** (Insert image reference here)

### 2. Decision Tree
A decision tree classifier splits data into subgroups based on feature importance.

#### Performance:
- Decision Tree Accuracy: **25.01%**
- Random Forest Accuracy: **43.61%**
- Random Forest with Grid Search CV: **44.56%**

#### Performance Metrics:
- **Confusion Matrix** (Insert image reference here)
- **F1-Score** (Insert image reference here)

## Accuracy vs. Number of Attributes
Using SVM coefficients, we visualize the importance of attributes in classification. (Insert image reference here)

## Challenges
- Handling **200** Bird Classes
- Feature Selection and Attribute Understanding
- Confusion Matrix Readability
- Understanding Different Machine Learning Models

## Conclusion
- The accuracy of both classification models is **less than 50%** due to dataset imbalance.
- Both models face difficulties in classifying the same set of bird species.

## What We Learned
- Machine Learning Techniques
- Python Programming
- Model Explainability and Feature Importance
- Team Collaboration

---
**Authors:** Zeeshan Ahmed, M. Uzair Janjua, Qasim Khalil  
**Date:** January 30, 2025  
**Affiliation:** xAI LAB Bamberg

