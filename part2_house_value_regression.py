import torch
import torch.utils.data
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Regressor():
    def __init__(self, x, nb_epoch=100, hidden_size=32, learning_rate=0.001, batch_size=16):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data.
            - nb_epoch {int} -- Number of epochs to train the network.
            - hidden_size {int} -- Number of neurons in hidden layers.
            - learning_rate {float} -- Learning rate for the optimizer.
            - batch_size {int} -- Batch size for training.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        try:
            X, _ = self._preprocessor(x, training=True)
            self.input_size = X.shape[1]
        except Exception as e:
            print(f"Preprocessor Error: {e}")
            raise ValueError("Issue with preprocessor function")

        self.output_size = 1

        self.nb_epoch = nb_epoch
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Define a 4-layer deep neural network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),  # Replacing Sigmoid
            torch.nn.Linear(self.hidden_size, self.output_size)
        ).to(device)  # Move model to GPU

        # Mean Squared Error loss for regression
        self.loss_criterion = torch.nn.MSELoss()

        # Adam optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of size (batch_size, 1).
        """

        #######################################################################
        #                        ** START OF YOUR CODE **
        #######################################################################

        # Work on a copy to avoid modifying the original data.
        x = x.copy()

        # Identify numeric and categorical columns.
        num_cols = x.select_dtypes(include=['number']).columns.tolist()
        cat_cols = x.select_dtypes(exclude=['number']).columns.tolist()

        # Fill missing values (if any).
        if training:
            self._num_means = x[num_cols].mean()
            self._num_stds = x[num_cols].std().replace(0, 1)  # Avoid division by zero

        if num_cols:
            x[num_cols] = x[num_cols].fillna(self._num_means)
        if cat_cols:
            x[cat_cols] = x[cat_cols].fillna("missing")

        # One-hot encode categorical columns.
        if cat_cols:
            x = pd.get_dummies(x, columns=cat_cols, drop_first=False)

        # Ensure consistency in training/testing column order
        if training:
            self._x_columns = x.columns  # Store training columns
        else:
            if hasattr(self, '_x_columns'):
                missing_cols = set(self._x_columns) - set(x.columns)
                for col in missing_cols:
                    x[col] = 0  # Add missing columns with zero values
                x = x[self._x_columns]  # Ensure column order matches training

        # Compute and apply Z-score normalisation for numeric columns
        x[num_cols] = (x[num_cols] - self._num_means) / self._num_stds

        # Convert boolean columns to float (Fix for PyTorch tensor conversion issue)
        bool_cols = x.select_dtypes(include=['bool']).columns.tolist()
        x[bool_cols] = x[bool_cols].astype(float)

        # Convert processed DataFrame to a torch tensor.
        X_tensor = torch.tensor(x.values, dtype=torch.float32)

        # Process target y if provided.
        if y is not None:
            y = y.copy()

            if training:
                self._y_mean = y.mean().values[0]  # Store as scalar
                self._y_std = y.std().replace(0, 1).values[0]  # Store as scalar
            else:
                # Ensure self._y_mean and self._y_std were already set during training
                if not hasattr(self, '_y_mean') or not hasattr(self, '_y_std'):
                    raise AttributeError("self._y_mean and self._y_std are missing. "
                                         "Ensure that fit() was called before predict().")

            y = y.fillna(self._y_mean)  # Fill missing values with scalar mean

            # Normalize y using the stored mean and std
            y_normalised = (y.values - self._y_mean) / self._y_std

            # Convert to PyTorch tensor
            Y_tensor = torch.tensor(y_normalised, dtype=torch.float32).reshape(-1, 1)
        else:
            Y_tensor = None

        # Return preprocessed x and y, return None for y if it was None
        return X_tensor, Y_tensor

        #######################################################################
        #                        ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the data. Ensure that X and Y are PyTorch tensors
        X, Y = self._preprocessor(x, y=y, training=True)

        # Set model to training mode
        self.model.train()

        # Create a dataset and data loader for batching and shuffling.
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        loss_history = []
        for epoch in range(self.nb_epoch):
            epoch_loss = 0.0

            # Iterate over batches.
            for batch_X, batch_Y in dataloader:

                self.optimiser.zero_grad()  # Reset gradients
                predictions = self.model(batch_X)  # Perform forward pass
                loss = self.loss_criterion(predictions, batch_Y)  # Compute the loss
                loss.backward()  # Perform backwards pass to compute gradients of loss
                self.optimiser.step()  # Perform one step of gradient descent and update weights
                epoch_loss += loss.item()  # Accumulate loss

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)

            # Print the average loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.nb_epoch}, Loss: {avg_loss:.4f}")

        # Plotting the loss
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.nb_epoch + 1), loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss During Training of the Neural Network Model')
        plt.show()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)

        # set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X)

        # Inverse the normalization to return predictions in the original scale.
        y_pred = predictions * self._y_std + self._y_mean
        return y_pred.cpu().numpy()

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        y_pred = self.predict(x)
        y_true = y.values

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        print(f"Root Mean Squared Error: {rmse:.2f}")

        return rmse

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(x_train, y_train):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Split the data into training and validation sets (80/20 split)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42
    )

    # Define the hyperparameter grid
    param_grid = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "hidden_size": [10, 20, 50],
        "batch_size": [16, 32, 64, 128],
        "nb_epoch": [100, 500, 1000, 1500]
    }

    best_mse, best_params = float('inf'), None

    total_combinations = (len(param_grid["learning_rate"]) *
                          len(param_grid["hidden_size"]) *
                          len(param_grid["batch_size"]) *
                          len(param_grid["nb_epoch"]))
    count = 0

    # Iterate over all combinations using itertools.product
    for lr, hs, bs, ep in product(param_grid["learning_rate"],
                                  param_grid["hidden_size"],
                                  param_grid["batch_size"],
                                  param_grid["nb_epoch"]):

        count += 1
        print(f"\nTesting combination {count}/{total_combinations}")
        print(f"Parameters: learning_rate={lr}, hidden_size={hs}, batch_size={bs}, nb_epoch={ep}")

        # Initialize and train the model on the training split
        model = Regressor(x_tr, nb_epoch=ep, hidden_size=hs, learning_rate=lr, batch_size=bs)
        model.fit(x_tr, y_tr)

        mse = model.score(x_val, y_val)
        print(f"Validation MSE: {mse:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_params = {
                "learning_rate": lr,
                "hidden_size": hs,
                "batch_size": bs,
                "nb_epoch": ep
            }

        print(f"Validation MSE: {mse:.4f}")

    print(f"\nBest hyperparameters: {best_params} with MSE: {best_mse:.4f}")
    return best_params

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def example_main():
    """
    Main function to train and evaluate the regressor.
    """

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    x = data.drop(columns=[output_label])
    y = data[[output_label]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    print("\nTraining model with default hyperparameters...\n")
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    loaded_regressor = load_regressor()
    final_error = loaded_regressor.score(x_test, y_test)
    print(f"\nFinal Regressor RMSE: {final_error:.2f}")

    # optimised_regressor = Regressor(x_train, learning_rate=learning_rate, hidden_size=hidden_size, batch_size=batch_size, nb_epoch=nb_epoch)
    # optimised_regressor.fit(x_train, y_train)
    # save_regressor(optimised_regressor)

def save_regressor(model, filepath="regressor.pkl"):
    """ Saves the trained regressor model. """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_regressor(filepath="regressor.pkl"):
    """ Loads a saved regressor model. """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    example_main()