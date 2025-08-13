#                                                                         به نام خدا
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# بارگیری داده‌ها
data = load_iris()
X, y = data.data, data.target

# تقسیم داده به Train (70%) و Test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# مدل با معیار Gini
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
model_gini.fit(X, y)

# مدل با معیار Entropy
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model_entropy.fit(X, y)

# رسم درخت Gini
plt.figure(figsize=(12, 6))
plot_tree(model_gini, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree (Gini)")
plt.show()

# رسم درخت Entropy
plt.figure(figsize=(12, 6))
plot_tree(model_entropy, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree (Entropy)")
plt.show()

# پیش‌بینی با هر دو مدل
y_pred_gini = model_gini.predict(X_test)
y_pred_entropy = model_entropy.predict(X_test)

# دقت (Accuracy) هر مدل
accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

print(f"Accuracy (Gini): {accuracy_gini:.2f}")
print(f"Accuracy (Entropy): {accuracy_entropy:.2f}")

# گزارش طبقه‌بندی (Classification Report)
print("\nClassification Report (Gini):")
print(classification_report(y_test, y_pred_gini, target_names=data.target_names))

print("\nClassification Report (Entropy):")
print(classification_report(y_test, y_pred_entropy, target_names=data.target_names))