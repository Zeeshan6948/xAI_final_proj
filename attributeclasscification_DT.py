# Import libraries for data manipulation and Decision Tree
import pandas as pd  # For handling data files
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing sets
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.metrics import accuracy_score  # Evaluating the model

# Qasim Khalil

# Load the dataset files
images = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\images.txt', sep=' ', header=None, names=['image_id', 'image_name'])
labels = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\image_class_labels.txt', sep=' ', header=None, names=['image_id', 'class_id'])
classes = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\classes.txt', sep=' ', header=None, names=['class_id', 'class_name'])
attributes = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\attributes.txt', sep=' ', header=None, names=['attribute_id', 'attribute_name'])
image_attributes = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\attributes\image_attribute_labels.txt', sep=' ', header=None,
                               usecols=[0, 1, 2], names=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
train_test_split_file = pd.read_csv(r'C:\Users\Zeeshan Ahmed\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011\train_test_split.txt', sep=' ', header=None,
                                    names=['image_id', 'is_training_image'])

# Merge the data to create a complete dataset
# Step 1: Merge images with class labels
data = pd.merge(images, labels, on='image_id')

# Step 2: Merge attributes into the dataset
attribute_data = image_attributes.pivot(index='image_id', columns='attribute_id', values='is_present')
attribute_data.fillna(0, inplace=True)  # Replace NaN values with 0
data = pd.merge(data, attribute_data, on='image_id')

# Step 3: Add train/test split
data = pd.merge(data, train_test_split_file, on='image_id')

# Check the dataset
print(data.head())

# Zeeshan Ahmed

# Separate features (attributes) and labels (class_id)
X = data.iloc[:, 3:-2]  # Attributes start from the 4th column, exclude `image_id` and `is_training_image`
y = data['class_id']  # Class labels

# Split the data into training and testing sets
X_train = X[data['is_training_image'] == 1]  # Training data
y_train = y[data['is_training_image'] == 1]
X_test = X[data['is_training_image'] == 0]  # Testing data
y_test = y[data['is_training_image'] == 0]

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=50, min_samples_leaf=1, min_samples_split=2)

# Train the model
dt_classifier.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# from sklearn.decomposition import PCA

# # Apply PCA to reduce features
# pca = PCA(n_components=100)  # Reduce to 100 principal components
# X_train_reduced = pca.fit_transform(X_train)
# X_test_reduced = pca.transform(X_test)

# print(f"Reduced Training set shape: {X_train_reduced.shape}, Reduced Testing set shape: {X_test_reduced.shape}")
# # Retrain Decision Tree with reduced features
# dt_classifier.fit(X_train_reduced, y_train)

# # Predict and evaluate accuracy
# y_pred_reduced = dt_classifier.predict(X_test_reduced)
# accuracy_reduced = accuracy_score(y_test, y_pred_reduced)
# print(f"Model Accuracy after PCA: {accuracy_reduced * 100:.2f}%")

# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [10, 20, 50, None],  # Limit depth to prevent overfitting
#     'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
#     'min_samples_leaf': [1, 5, 10]  # Minimum samples per leaf
# }

# # Initialize GridSearchCV
# grid_search = GridSearchCV(
#     DecisionTreeClassifier(random_state=42),
#     param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='accuracy',
#     verbose=1,
#     n_jobs=-1
# )

# # Perform grid search on the original training data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest with some default parameters
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest on the original training data
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")

# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid for tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],          # Number of trees in the forest
#     'max_depth': [10, 20, None],             # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],         # Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4],           # Minimum samples required at a leaf node
#     'bootstrap': [True, False]               # Whether to use bootstrap samples
# }

# # Initialize GridSearchCV with Random Forest
# grid_search = GridSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring='accuracy',
#     verbose=1,
#     n_jobs=-1
# )

# # Perform the grid search on the reduced feature set
# grid_search.fit(X_train, y_train)

# # Get the best parameters and score
# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")
# Initialize Random Forest with some default parameters
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42 ,bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2)

# Train the Random Forest on the original training data
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Convert the confusion matrix into a DataFrame for better readability
conf_matrix_df = pd.DataFrame(conf_matrix, index=classes['class_name'], columns=classes['class_name'])

# Sum up all misclassified cases for each class (row-wise sum, excluding diagonal)
misclassifications = conf_matrix_df.copy()
np.fill_diagonal(misclassifications.values, 0)  # Remove correct classifications
misclassified_counts = misclassifications.sum(axis=1)  # Total misclassifications per class

# Get the top 20 most misclassified classes
top_20_confused = misclassified_counts.nlargest(20).index
conf_matrix_top_20 = conf_matrix_df.loc[top_20_confused, top_20_confused]
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_top_20, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Top 20 Most Confused Bird Classes")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()



from sklearn.metrics import classification_report
# Get F1 scores for each class
report = classification_report(y_test, y_pred_rf, output_dict=True)

# Extract F1 scores for each class
f1_scores = {int(k): v['f1-score'] for k, v in report.items() if k.isdigit()}  # Convert keys to int

# Plot F1 Scores
plt.figure(figsize=(12, 6))
plt.bar(f1_scores.keys(), f1_scores.values(), color='blue')
plt.xlabel("Class ID")
plt.ylabel("F1 Score")
plt.title("F1 Score for Each Class - Decision Tree")
plt.show()


# Muhammad Uzair Janjua

from sklearn.metrics import classification_report

# Get F1 scores for each class
report = classification_report(y_test, y_pred_rf, output_dict=True)
f1_scores = {int(k): v["f1-score"] for k, v in report.items() if k.isdigit()}

# Convert to DataFrame
f1_df = pd.DataFrame(list(f1_scores.items()), columns=["class_id", "f1_score"])

# Get top 10 highest and lowest F1-score classes
top_10_highest = f1_df.nlargest(10, "f1_score")
top_10_lowest = f1_df.nsmallest(10, "f1_score")

# Combine both highest and lowest F1-score classes
combined_f1 = pd.concat([top_10_lowest, top_10_highest])

# Add class names for visualization
combined_f1["class_name"] = combined_f1["class_id"].map(classes["class_name"])

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x="f1_score", y="class_name", data=combined_f1, palette=["red"] * 10 + ["green"] * 10)
plt.xlabel("F1 Score")
plt.ylabel("Bird Class Name")
plt.title("Top 10 Lowest and Highest F1 Score Classes")
plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold")
plt.legend()
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Get feature importances from the Random Forest model
importances = rf_classifier.feature_importances_

# Sort the features based on their importance
indices = np.argsort(importances)[::-1]  # Get indices of features sorted by importance in descending order

# Print the top 10 features and their importance
print("Top 10 Features and their Importance:")
for i in range(10):
    print(f"{i + 1}. Feature {indices[i]}: Importance {importances[indices[i]]:.4f}")
# Initialize the list to store accuracies
accuracies = []

# Create different subsets of features (increasing number of features)
for num_features in range(250, 260):  # Loop through from 1 feature to all 311 features
    # Select the top 'num_features' from the sorted list
    X_train_subset = X_train.iloc[:, indices[:num_features]]
    X_test_subset = X_test.iloc[:, indices[:num_features]]
    
    # Initialize and train a Decision Tree Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42 ,bootstrap=False, max_depth=20, min_samples_leaf=1, min_samples_split=2)
    rf_classifier.fit(X_train_subset, y_train)
    
    # Predict on the test data
    y_pred = rf_classifier.predict(X_test_subset)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print(f"Accuracies for different feature subsets: {accuracies[:10]}")  # Print the first 10 accuracies for inspection
import matplotlib.pyplot as plt

# Plot the accuracy vs number of features
plt.figure(figsize=(10, 6))
plt.plot(range(250, 260), accuracies, marker='o', color='b', label='Decision Tree Accuracy')
plt.title('Accuracy vs. Number of Features for Decision Tree')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

