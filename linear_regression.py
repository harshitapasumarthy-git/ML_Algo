import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.0001, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_values=[]

    def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
        
            for _ in range(self.n_iterations):
                linear_model = np.dot(X, self.weights) + self.bias
                cost = np.sum([data**2 for data in (y-linear_model)]) / len(y)

                self.cost_values.append(cost)

                # Compute gradients
                d_weights = np.dot(X.T, (linear_model - y)) / n_samples
                d_bias = np.sum(linear_model - y) / n_samples

                # Update parameters
                self.weights -= self.learning_rate * d_weights
                self.bias -= self.learning_rate * d_bias


            return self.weights, self.bias


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression with Gradient Descent")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of iterations for gradient descent")
    args = parser.parse_args()
    

    if args.data:
        # Load data
        data = np.genfromtxt(args.data, delimiter=',')
        X = data[1:24, :-1]
        y = data[1:24, -1]
        X_test = data[25:, :-1]
        y_test = data[25:, -1]
        # Initialize and train the model
        model = LinearRegressionGradientDescent(learning_rate=args.learning_rate,
                                                 n_iterations=args.n_iterations,)
        model.fit(X, y)
       


        # Make predictions
        predictions = model.predict(X_test)
        x_line = np.linspace(min(X), max(X), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        print("Predictions:", predictions)
        mse = mean_squared_error(y_test, predictions)
        print("Mean squared error:", mse)


        # Plot the training data points
        plt.scatter(X, y, label='Training Data')

        # Plot the linear regression line
        plt.plot(x_line, y_line, color='red', label='Linear Regression Line')
        

        # Add labels and legend
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Linear Regression Line')
        plt.legend()
        # Show the plot
        plt.show()
        plt.plot(range(len(model.cost_values)), model.cost_values,'-r')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Iterations vs Cost')
        plt.show()
        
    else:
        print("Please provide the path to the data file using the '--data' argument.")
