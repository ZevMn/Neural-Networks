import torch
import torch.utils.data
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class Regressor():

    def __init__(self, x, nb_epoch = 1000, hidden_size=10, learning_rate = 0.01, batch_size = 128):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Define a simple neural network architecture using PyTorch
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

        # Mean Squared Error loss for regression
        self.loss_criterion = torch.nn.MSELoss()

        # Adam optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Work on a copy to avoid modifying the original data.
        x = x.copy()

        # Identify numeric and categorical columns.
        num_cols = x.select_dtypes(include=['number']).columns.tolist()
        cat_cols = x.select_dtypes(exclude=['number']).columns.tolist()

        # Fill missing values (if any).
        if training:
            self._num_means = x[num_cols].mean()
            self._num_stds = x[num_cols].std().replace(0, 1) # Avoid division by zero

        if num_cols:
            x[num_cols] = x[num_cols].fillna(self._num_means)
        if cat_cols:
            x[cat_cols] = x[cat_cols].fillna("missing")

        # One-hot encode categorical columns.
        x = pd.get_dummies(x, columns=cat_cols, drop_first=False)

        if training:
            # Store the full set of columns to ensure consistency in test mode.
            self._x_columns = x.columns
        else:
            # Reindex to match training columns, if stored.
            if hasattr(self, '_x_columns'):
                x = x.reindex(columns=self._x_columns, fill_value=0)

        # Compute and apply Z-score normalisation for numeric columns
        x[num_cols] = (x[num_cols] - self._num_means) / self._num_stds

        # Convert processed DataFrame to a torch tensor.
        X_tensor = torch.tensor(x.values, dtype=torch.float32)

        # Process target y if provided.
        if y is not None:
            y = y.copy()

            if training:
                self._y_mean = y.mean()
                self._y_std = y.std().replace(0, 1)

            y = y.fillna(self._y_mean) # Fill in missing y values with training mean

            y_normalised = (y.values - self._y_mean) / self._y_std # Normalise
            Y_tensor = torch.tensor(y_normalised, dtype=torch.float32).reshape(-1, 1)

        else:
            Y_tensor = None

        # Return preprocessed x and y, return None for y if it was None
        return X_tensor, Y_tensor

        #######################################################################
        #                       ** END OF YOUR CODE **
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
        X, Y = self._preprocessor(x, y=y, training=False)

        # Set model to training mode
        self.model.train()

        # Create a dataset and data loader for batching and shuffling.
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
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

            # Print the average loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.nb_epoch}, Loss: {avg_loss:.4f}")

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
        return y_pred.numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        self.model.eval()

        y_pred = self.predict(x)
        #y_true = Y.numpy() ???
        y_true = y.values
        y_true = y.values if isinstance(y, pd.DataFrame) else np.array(y)

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        ### r2 = r2_score(y_true, y_pred)

        print(f"Mean Squared Error: {mse:.2f}")
        ### print(f"R² Score: {r2:.4f}")

        return mse

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



def perform_hyperparameter_search(): 
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

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()

