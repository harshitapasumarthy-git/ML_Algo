import numpy as np
import argparse
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
    return 1/(1+np.exp(-z))
# def calculate_accuracy(y_true, y_pred):
#         correct_predictions = np.sum(y_true == y_pred)
#         total_samples = len(y_true)
#         accuracy = correct_predictions / total_samples
#         return accuracy

def logistic_regression_newton(X, y,learning_rate=0.01, n_iterations=1000, tol=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X,weights)+bias
        y_predicted = sigmoid(linear_model)
        
        # Compute gradient
        gradient = np.dot(X.T, (y_predicted - y))/n_samples

        # Compute Hessian matrix
        hessian = np.dot(X.T, np.dot(np.diagflat(y_predicted * (1 - y_predicted)), X)) / n_samples 

        # Update parameters using Newton's method
        try:
            update_weights = np.linalg.inv(hessian).dot(gradient)
        except np.linalg.LinAlgError:
            break
        old_weights = weights.copy()
        weights -= learning_rate * update_weights
        update_bias = np.sum(y_predicted - y) / n_samples
        bias -= learning_rate * update_bias
        if np.linalg.norm(weights - old_weights) < tol:
            break


    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)
   

def logistic_regression_gradient_ascent(X, y,learning_rate=0.001, n_iterations=1000, tol=0.00001):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X,weights)+bias
        y_predicted = 1/(1+np.exp(-linear_model))

        # Compute gradient
        gradient = np.dot(X.T, (y_predicted - y)) / n_samples

        # Update parameters using gradient ascent
        # '''complete code here'''
        old_weights = weights.copy()
        weights -= learning_rate * gradient
        bias -= learning_rate * np.sum(y_predicted - y) / n_samples
        
        if np.linalg.norm(weights - old_weights) < tol:
            break
        linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]

    return np.array(y_predicted_cls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method or Gradient Ascent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--test_data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--method", type=str, default="newton", choices=["newton", "gradient"], help="Optimization method (newton or gradient)")
   
    args = parser.parse_args()

    if args.data:
        data = np.genfromtxt(args.data, delimiter=',')
        test_data = np.genfromtxt(args.test_data, delimiter=',')

        X = data[1:, :-1]
        y = data[1:, -1]
        X_test = test_data[1:, :-1]
        y_test = test_data[1:, -1]
        if args.method == "newton":
            predictions =logistic_regression_newton(X_test, y_test)
            accuracy = accuracy_score(y_test, predictions)
 
               
        else :
            predictions   = logistic_regression_gradient_ascent(X_test, y_test)
            accuracy = accuracy_score(y_test,predictions)
        

        print("Predictions:", predictions)
        print("Accuracy",accuracy)
        cm = confusion_matrix(y_test, predictions)
     # Display the confusion matrix
        print("Confusion Matrix:")
        print(cm)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    
    else:
        print("Please provide the path to the training data file using the '--data' argument and testing data file using the '--test_data' argument.")
