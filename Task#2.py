# Import libraries
from sklearn import datasets
import matplotlib
matplotlib.use('TkAgg')  # Important for VS Code to show plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load the dataset
digits = datasets.load_digits()

# Check the available keys in the dataset
print("Keys in the dataset:", digits.keys())

# 2. Define all keys briefly
for key in digits.keys():
    print(f"\n{key}:")
    if key == 'DESCR':
        print(digits[key][:300] + "...\n")  # Shortened description for readability
    else:
        print(digits[key], "\n")

# 3. Check the shape of the dataset
X = digits.data
y = digits.target

print("Shape of features (X):", X.shape)    # (1797, 64)
print("Shape of labels (y):", y.shape)       # (1797,)

# 4. Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 5. KNN Model and finding the best K
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    print(f"K = {k}, Accuracy = {acc:.4f}")

# Plot accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title('K vs Accuracy')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# 6. Best K
best_k = k_values[accuracies.index(max(accuracies))]
print(f"\n✅ Best value of K is: {best_k} with Accuracy = {max(accuracies)*100:.2f}%")

# 7. Train final model with best K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# 8. Compute accuracy on test data
accuracy = knn.score(X_test, y_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")

# 9. Plot confusion matrix and classification report
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 10. Evaluate model: Overfitting/Underfitting
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

if train_accuracy > test_accuracy:
    print("\nModel may be slightly overfitting (training accuracy is higher).")
elif train_accuracy < test_accuracy:
    print("\nModel may be underfitting.")
else:
    print("\nModel is well generalized.")
