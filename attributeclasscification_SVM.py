# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Path to the folder containing your dataset files
dataset_path = r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011'

# Load images.txt - Image ID and Image Name
images_df = pd.read_csv(os.path.join(dataset_path, r'CUB_200_2011\images.txt'), sep=r'\s+', header=None, names=['image_id', 'image_name'])

# Load image_class_labels.txt - Image ID and Class ID
image_class_labels_df = pd.read_csv(os.path.join(dataset_path, r'CUB_200_2011\image_class_labels.txt'), sep=r'\s+', header=None, names=['image_id', 'class_id'])

# Load classes.txt - Class ID and Class Name
classes_df = pd.read_csv(os.path.join(dataset_path, r'CUB_200_2011\classes.txt'), sep=r'\s+', header=None, names=['class_id', 'class_name'])

# Load attributes.txt - Attribute ID and Attribute Name
attributes_df = pd.read_csv(os.path.join(dataset_path, 'attributes.txt'), sep=r'\s+', header=None, names=['attribute_id', 'attribute_name'])

# Load image_attribute_labels.txt but only keep the first three columns
image_attribute_labels_df = pd.read_csv(
    os.path.join(dataset_path, r'CUB_200_2011\attributes\image_attribute_labels.txt'), 
    delim_whitespace=True, header=None, 
    names=['image_id', 'attribute_id', 'is_present'], usecols=[0, 1, 2]
)

# Load train_test_split.txt - Image ID and Training Flag
train_test_split_df = pd.read_csv(os.path.join(dataset_path, r'CUB_200_2011\train_test_split.txt'), sep=r'\s+', header=None, names=['image_id', 'is_training_image'])

# Merge the image_class_labels_df and image_attribute_labels_df to create the feature matrix
merged_df = pd.merge(image_attribute_labels_df, image_class_labels_df, on='image_id')

# Creating the feature matrix X (binary features for attributes) and label vector y (class IDs)
num_images = len(images_df)
num_attributes = len(attributes_df)

# Initialize the feature matrix X and label vector y
X = np.zeros((num_images, num_attributes))  # Each row will represent one image, each column represents an attribute
y = np.zeros(num_images)

# Fill the feature matrix X with attribute presence information (0 or 1)
for index, row in merged_df.iterrows():
    image_idx = row['image_id'] - 1  # Make image_id zero-based
    attribute_idx = row['attribute_id'] - 1  # Make attribute_id zero-based
    X[image_idx, attribute_idx] = row['is_present']  # Set the value based on attribute presence
    y[image_idx] = row['class_id'] - 1  # Set the class label for the image (zero-based class id)

# Split the data into training and test sets based on train_test_split.txt
train_images = train_test_split_df[train_test_split_df['is_training_image'] == 1]['image_id'] - 1
test_images = train_test_split_df[train_test_split_df['is_training_image'] == 0]['image_id'] - 1

# Create train and test sets
X_train = X[train_images]
y_train = y[train_images]
X_test = X[test_images]
y_test = y[test_images]

# Create and train an SVM model (Linear SVM)
# svm_model = svm.SVC(kernel='linear')  # Using a linear kernel
# svm_model = svm.SVC(kernel='rbf', gamma='scale')  # Using RBF kernel
svm_model = svm.SVC(kernel='linear')  # Using RBF kernel, Tune C and gamma
svm_model.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plotting the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_df['class_name'], yticklabels=classes_df['class_name'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

############## Experiment with a grid search to find the best parameters for your model using GridSearchCV from sklearn:
# from sklearn.model_selection import GridSearchCV
# parameter_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['linear', 'rbf']}
# grid_search = GridSearchCV(svm.SVC(), parameter_grid, cv=3)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import numpy as np

# # Compute the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Get the sum of misclassifications for each class
# misclassifications = np.sum(cm, axis=1) - np.diag(cm)

# # Get the indices of the top 20 most confused classes
# top_20_confused_classes = np.argsort(misclassifications)[20:]  # Get top 20

# # Extract the corresponding confusion matrix rows/columns
# cm_top_20 = cm[top_20_confused_classes][:, top_20_confused_classes]

# # Get class names for the top 20 confused classes
# top_20_class_names = classes_df['class_name'].iloc[top_20_confused_classes].values

# # Plot the confusion matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(cm_top_20, annot=True, fmt='d', cmap='Blues', xticklabels=top_20_class_names, yticklabels=top_20_class_names)
# plt.xlabel('Predicted Class')
# plt.ylabel('True Class')
# plt.title('Confusion Matrix (Top 20 Most Confused Classes)')
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.show()

# from sklearn.metrics import classification_report

# # Generate the classification report
# report = classification_report(y_test, y_pred, target_names=classes_df['class_name'])

# # Print the report
# print(report)

# from sklearn.metrics import precision_recall_fscore_support
# import pandas as pd

# # Compute precision, recall, and F1-score
# precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# # Create a DataFrame with class names and F1-scores
# f1_df = pd.DataFrame({'Class': classes_df['class_name'], 'F1-Score': f1})

# # Sort by F1-score
# top_10_highest = f1_df.nlargest(10, 'F1-Score')  # Top 10 highest F1-score
# top_10_lowest = f1_df.nsmallest(10, 'F1-Score')  # Top 10 lowest F1-score

# # Combine both for a single visualization
# f1_combined = pd.concat([top_10_lowest, top_10_highest])

# # Bar chart
# plt.figure(figsize=(12, 6))
# plt.barh(f1_combined['Class'], f1_combined['F1-Score'], color=['red'] * 10 + ['green'] * 10)
# plt.xlabel('F1-Score')
# plt.ylabel('Class Name')
# plt.title('Top 10 Lowest & Highest F1-Score Classes')
# plt.axvline(x=np.mean(f1), color='black', linestyle='dashed', label='Mean F1-Score')  # Mean F1-Score
# plt.legend()
# plt.gca().invert_yaxis()  # Invert to have the best at the top
# plt.show()


# import numpy as np

# # Sum feature presence across all images
# feature_importance = np.sum(X_train, axis=0)  

# # Create a DataFrame
# feature_df = pd.DataFrame({'Attribute': attributes_df['attribute_name'], 'Importance': feature_importance})

# # Get the top 10 most important attributes
# top_10_attributes = feature_df.nlargest(10, 'Importance')

# # Plot feature importance for top 10 attributes
# plt.figure(figsize=(12, 6))
# plt.bar(top_10_attributes['Attribute'], top_10_attributes['Importance'], color='teal')
# plt.xticks(rotation=90)
# plt.xlabel('Attribute Name')
# plt.ylabel('Importance (Sum of Presence)')
# plt.title('Top 10 Most Important Attributes')
# plt.show()


# import matplotlib.pyplot as plt

# # Count images per class
# class_counts = image_class_labels_df['class_id'].value_counts()

# # Count how many classes have the same number of images
# most_common_count = class_counts.value_counts().idxmax()  # Most frequent image count
# same_count_classes = np.sum(class_counts == most_common_count)
# different_count_classes = len(class_counts) - same_count_classes

# # Compute percentages
# total_classes = len(class_counts)
# same_percentage = (same_count_classes / total_classes) * 100
# different_percentage = (different_count_classes / total_classes) * 100

# # Pie chart data
# labels = ['Same Image Count', 'Different Image Count']
# sizes = [same_percentage, different_percentage]
# colors = ['green', 'red']

# # Plot pie chart
# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
# plt.title('Class Distribution: Same vs. Different Image Counts')
# plt.show()

# feature_importance = np.abs(svm_model.coef_).mean(axis=0)
# accuracies = []
# for num_features in range(250, 260):  # Test with top 1 to 20 attributes
#     selected_features = np.argsort(-feature_importance)[:num_features]
#     X_train_selected = X_train[:, selected_features]
#     X_test_selected = X_test[:, selected_features]
    
#     # Train and evaluate the SVM model
#     temp_model = svm.SVC(kernel='linear')
#     temp_model.fit(X_train_selected, y_train)
#     y_pred_temp = temp_model.predict(X_test_selected)
#     acc = accuracy_score(y_test, y_pred_temp)
#     accuracies.append(acc)

# # Plot accuracy vs. number of attributes
# plt.figure(figsize=(10, 6))
# plt.plot(range(250, 260), accuracies, marker='o')
# plt.xlabel('Number of Attributes Used')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Number of Attributes')
# plt.grid()
# plt.show()


import matplotlib.pyplot as plt

# Accuracy values for different models (example)
model_names = ['SVM (Linear)', 'SVM (RBF)', 'SVM (RBF) + GridSearchCV']
accuracies = [43.87, 45.63, 46.67]  # Example accuracies

# Bar chart
plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel('Model Type')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0, 100)
plt.show()