import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("student_education_dataset.csv")


num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())


cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])



X = df.drop("StudentPerformance", axis=1)
y = df["StudentPerformance"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_split=5,
    random_state=42
)



cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("\n5-Fold CV Accuracy:", round(cv_scores.mean(), 3))


model.fit(X_train, y_train)



test_accuracy = model.score(X_test, y_test)
print("Test Accuracy:", round(test_accuracy, 3))

y_pred = model.predict(X_test)


def generate_learning_path(label):
    if label == 0:
        return "Beginner Path: Basics, Programming Fundamentals, Math Foundation"
    elif label == 1:
        return "Intermediate Path: DSA, OOPS, DBMS, Mini Projects"
    else:
        return "Advanced Path: AI, Machine Learning, Internships, Competitive Coding"


print("\nPERSONALIZED LEARNING PATH OUTPUT:\n")
for i in range(10):
    print(f"Student {i+1}: Class = {y_pred[i]} → {generate_learning_path(y_pred[i])}")



print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX:\n", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Beginner","Intermediate","Advanced"],
            yticklabels=["Beginner","Intermediate","Advanced"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()



plt.figure(figsize=(28,16))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Beginner", "Intermediate", "Advanced"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Output - Personalized Learning Path")
plt.savefig("decision_tree_output.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDECISION TREE RULES (TEXT FORMAT):\n")

tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

print("\nSaved Files:")
print("decision_tree_output.png")
print("confusion_matrix.png")
print("decision_tree_rules.txt")