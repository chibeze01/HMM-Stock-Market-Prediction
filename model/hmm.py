import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMMStockPredictor:
    """
    A class for a Hidden Markov Model (HMM) based stock market predictor.
    """
    def __init__(self, n_hidden_states=4, n_iter=1000):
        """
        Initializes the HMMStockPredictor.

        Parameters:
        - n_hidden_states (int): The number of hidden states in the HMM.
        - n_iter (int): The number of iterations to perform during training.
        """
        self.n_hidden_states = n_hidden_states
        self.n_iter = n_iter
        self.model = GaussianHMM(n_components=self.n_hidden_states,
                                 covariance_type="diag",
                                 n_iter=self.n_iter)

    def train(self, X_train):
        """
        Trains the HMM model on the provided training data.

        Parameters:
        - X_train (array-like): The training data (sequence of observations).
        """
        self.model.fit(X_train)

    def fine_tune(self, X_new):
        """
        Fine-tunes the HMM model on new data. For this implementation,
        it's equivalent to retraining the model on the new data.

        Parameters:
        - X_new (array-like): The new data for fine-tuning.
        """
        # In a more complex scenario, fine-tuning might involve adjusting
        # parameters without a full retraining, but for this HMM,
        # re-fitting is a straightforward approach.
        self.model.fit(X_new)

    def predict_next_day_state(self, X):
        """
        Predicts the most likely hidden state for the next day.

        Parameters:
        - X (array-like): The sequence of observations leading up to the prediction.

        Returns:
        - int: The predicted hidden state for the next day.
        """
        # Predict the sequence of hidden states for the given observations
        hidden_states = self.model.predict(X)

        # Get the last hidden state in the sequence
        last_hidden_state = hidden_states[-1]

        # Use the transition matrix to find the most probable next state
        # The transition matrix `transmat_` gives the probability of
        # transitioning from one state to another.
        most_likely_next_state = np.argmax(self.model.transmat_[last_hidden_state])

        return most_likely_next_state