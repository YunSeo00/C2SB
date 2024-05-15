import numpy as np

class PMF:
    def __init__(self, num_factors, learning_rate=0.01, regularization=0.1, num_epochs=100, sigma2=0.1, sigma_UV=0.1, batch_size=1000, tol=1e-6, patience=10, seed=42):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.sigma2 = sigma2
        self.sigma_UV = sigma_UV
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        np.random.seed(seed)

    def fit(self, R):
        """
        Fit the PMF model to the given rating matrix R.
        
        Parameters:
        R (numpy.ndarray): The rating matrix to be factorized.
        """
        self.num_users, self.num_items = R.shape
        self.R = R / np.max(R)  # Scale ratings to 0-1
        
        # Initialize user and item latent factor matrices
        self.U = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        self.V = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors))
        
        # Initialize velocity terms for momentum
        self.U_velocity = np.zeros_like(self.U)
        self.V_velocity = np.zeros_like(self.V)

        # Get indices of non-zero ratings
        self.non_zero_indices = np.array([(i, j) for i in range(self.num_users) for j in range(self.num_items) if self.R[i, j] > 0])
        
        best_rmse = float('inf')
        best_U = None
        best_V = None
        no_improvement_count = 0
        
        # Training using Stochastic Gradient Descent (SGD) with batch processing and early stopping
        for epoch in range(self.num_epochs):
            np.random.shuffle(self.non_zero_indices)
            batches = [self.non_zero_indices[k:k+self.batch_size] for k in range(0, len(self.non_zero_indices), self.batch_size)]
            
            for batch in batches:
                self.sgd_step(batch)

            # Normalize user and item latent factor matrices
            self.normalize()

            train_rmse = self.rmse()
            print(f"Epoch: {epoch + 1}, RMSE: {train_rmse}")
            
            # for debugging
            if np.isnan(train_rmse):
                print("NaN RMSE detected. Printing debug information:")
                predictions = self.predict()
                print("Predictions:")
                print(predictions)
                print("User Latent Factors:")
                print(self.U)
                print("Item Latent Factors:")
                print(self.V)
                break
            
            if train_rmse < best_rmse:
                best_rmse = train_rmse
                best_U = self.U.copy()
                best_V = self.V.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                self.learning_rate *= 0.5 # decay learning rate by half if no improvement

            if train_rmse < self.tol:
                print("Tolerance met. Stopping early.")
                self.U = best_U
                self.V = best_V
                return
            
            if no_improvement_count >= self.patience:
                print("No improvement for 10 steps. Stopping early.")
                self.U = best_U
                self.V = best_V
                return
        
        # Restore best U and V
        self.U = best_U
        self.V = best_V

    def sgd_step(self, batch):
        """
        Perform a single SGD step on a given batch of data.
        
        Parameters:
        batch (list): List of tuples (user_index, item_index).
        """
        user_indices = [index[0] for index in batch]
        item_indices = [index[1] for index in batch]
        
        U_batch = self.U[user_indices, :]
        V_batch = self.V[item_indices, :]
        R_batch = self.R[user_indices, item_indices]
        
        predictions = self.sigmoid(np.sum(U_batch * V_batch, axis=1))
        errors = R_batch - predictions
        
        U_grad = (errors[:, np.newaxis] * V_batch - self.U[user_indices, :] / self.sigma_UV**2) / self.sigma2
        V_grad = (errors[:, np.newaxis] * U_batch - self.V[item_indices, :] / self.sigma_UV**2) / self.sigma2
        
        self.U[user_indices, :] -= self.learning_rate * U_grad
        self.V[item_indices, :] -= self.learning_rate * V_grad

    def sigmoid(self, x):
        """
        Apply the sigmoid function to the input array.
        
        Parameters:
        x (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Sigmoid of the input.
        """
        clipped_x = np.clip(x, -6, 6)  # Clipping to avoid overflow. return value is bounded (0.0024, 0.9975)
        return 1 / (1 + np.exp(-clipped_x)) 

    def normalize(self):
        """
        Normalize the user and item latent factor matrices so that their magnitudes are at most 1.
        """
        self.U = self.U / np.maximum(1, np.linalg.norm(self.U, axis=1)[:, np.newaxis])
        self.V = self.V / np.maximum(1, np.linalg.norm(self.V, axis=1)[:, np.newaxis])

    def predict(self):
        """
        Predict the full rating matrix using the learned user and item latent factor matrices.
        
        Returns:
        numpy.ndarray: The predicted rating matrix.
        """
        return self.sigmoid(np.dot(self.U, self.V.T))

    def predict_single(self, user, item):
        """
        Predict a single rating given a user and an item.
        
        Parameters:
        user (int): The user index.
        item (int): The item index.
        
        Returns:
        float: The predicted rating.
        """
        return self.sigmoid(np.dot(self.U[user, :], self.V[item, :]))

    def rmse(self):
        """
        Compute the root mean squared error on the training set.
        
        Returns:
        float: The RMSE value.
        """
        predictions = self.predict()
        mask = self.R > 0
        rmse = np.sqrt(np.sum((self.R[mask] - predictions[mask]) ** 2) / np.sum(mask))
        if np.isnan(rmse):
            print("NaN RMSE detected during computation.")
        return rmse
    
# Example usage
if __name__ == "__main__":
    # Sample rating matrix (0 indicates missing entries)
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    pmf = PMF(num_factors=3, learning_rate=0.0001, regularization=0.01, num_epochs=100, batch_size=2, patience=30)
    pmf.fit(R)
    predictions = pmf.predict()
    print("Predicted Rating Matrix:")
    print(predictions)
    print("item embedding matrix: ", pmf.V)
