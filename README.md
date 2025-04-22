# Predicting Online Course Completion

This project focuses on building a machine learning model to classify whether a learner will complete an online course based on their activity logs. The goal is to use basic learner engagement data (such as videos watched, assignments submitted, and forum participation) to predict course completion outcomes.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Code and Implementation](#code-and-implementation)
- [Model Evaluation](#model-evaluation)
- [Output Screenshot](#output-screenshot)
- [How to Run](#how-to-run)
- [References and Credits](#references-and-credits)

---

## Problem Statement

Online learning platforms face challenges with learner retention and course completion rates. Being able to predict which learners are likely to complete a course can enable platforms to take proactive steps to improve learner engagement and success rates. This project uses a classification approach to solve this problem.

---

## Technologies Used

- Python (3.x)
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Dataset Description

The dataset contains learner activity logs and includes the following columns:

- `videos_watched`: Number of video lectures watched
- `assignments_submitted`: Number of assignments submitted
- `forum_posts`: Number of posts made in discussion forums
- `completed`: Target label indicating course completion (yes/no)

The `completed` column is encoded into binary values:
- 1 = Completed
- 0 = Not Completed

---

## Methodology

1. **Data Preprocessing**
   - Encoding categorical target variable.
   - Splitting features and target variable.
   - Dividing data into training and test sets.

2. **Modeling**
   - A logistic regression model was trained to perform binary classification.

3. **Evaluation**
   - Confusion matrix was used for visual analysis.
   - Metrics such as Accuracy, Precision, and Recall were calculated.
   - ROC-AUC score and curve were also generated.

---

## Code and Implementation

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('/mnt/data/online_learning.csv')

# Encode the target variable
df['completed'] = df['completed'].map({'yes': 1, 'no': 0})

# Define features and target
X = df[['videos_watched', 'assignments_submitted', 'forum_posts']]
y = df['completed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Completed', 'Completed'],
            yticklabels=['Not Completed', 'Completed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
