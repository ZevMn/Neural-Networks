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
    def __init__(self, x, nb_epoch=100, hidden_size=32, learning_rate=0.01, batch_size=64):
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

        # Define a 3-layer deep neural network
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
        """

        #######################################################################
        #                        ** START OF YOUR CODE **
        #######################################################################

        x = x.copy()
        num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = x.select_dtypes(exclude=['number']).columns.tolist()

        if training:
            self._num_means = x[num_cols].mean()
            self._num_stds = x[num_cols].std().replace(0, 1)

        x[num_cols] = x[num_cols].fillna(self._num_means)
        x[cat_cols] = x[cat_cols].fillna("missing")

        if cat_cols:
            x = pd.get_dummies(x, columns=cat_cols, drop_first=False).astype(float)

        if training:
            self._x_columns = list(x.columns)
        else:
            x = x.reindex(columns=self._x_columns, fill_value=0)

        x[num_cols] = (x[num_cols] - self._num_means) / self._num_stds

        X_tensor = torch.tensor(x.values, dtype=torch.float32).to(device)

        if y is not None:
            y = y.copy()
            if training:
                self._y_mean = y.mean().values[0]
                self._y_std = max(y.std().values[0], 1e-8)

            y = y.fillna(self._y_mean).astype(float)
            y_normalised = (y.values - self._y_mean) / self._y_std
            Y_tensor = torch.tensor(y_normalised, dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            Y_tensor = None

        return X_tensor, Y_tensor

    def fit(self, x, y):
        """
        Regressor training function.
        """
        X, Y = self._preprocessor(x, y=y, training=True)

        self.model.train()
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_history = []
        for epoch in range(self.nb_epoch):
            epoch_loss = 0.0
            for batch_X, batch_Y in dataloader:
                self.optimiser.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_criterion(predictions, batch_Y)
                loss.backward()
                self.optimiser.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.nb_epoch}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, x):
        """
        Output the predicted value for a given input.
        """
        X, _ = self._preprocessor(x, training=False)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        y_pred = predictions * self._y_std + self._y_mean
        return y_pred.cpu().numpy()

    def score(self, x, y):
        """
        Evaluate the model accuracy.
        """
        y_pred = self.predict(x)
        y_true = y.values
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse:.2f}")
        return rmse


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