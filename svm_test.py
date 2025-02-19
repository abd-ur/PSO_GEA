from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train an SVM classifier
svm_without_pso = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_without_pso.fit(X_train, y_train)

# Test the classifier
y_pred_without_pso = svm_without_pso.predict(X_test)

# Evaluate performance
print("Performance of SVM without PSO:")
print(classification_report(y_test, y_pred_without_pso))
