"""LSTM Autoencoder anomaly detector for AnomalyShield."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.detector import BaseDetector


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    The encoder compresses an input sequence into a fixed-size hidden state.
    The decoder reconstructs the original sequence from that hidden state.
    High reconstruction error signals anomalous input.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
    hidden_dim : int
        Size of the LSTM hidden state.
    n_layers : int
        Number of stacked LSTM layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstruction of shape (batch, seq_len, input_dim).
        """
        # Encode: run the full sequence, take the last hidden state
        _, (hidden, cell) = self.encoder(x)

        seq_len = x.size(1)
        # Repeat the last hidden output as decoder input for each timestep
        # hidden shape: (n_layers, batch, hidden_dim) — take the last layer
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # Project back to input dimension
        reconstruction = self.output_layer(decoder_output)
        return reconstruction


class AutoencoderDetector(BaseDetector):
    """Anomaly detector using an LSTM autoencoder.

    Trains on normal data and flags samples with high reconstruction error
    as anomalies.

    Parameters
    ----------
    name : str
        Detector name for identification in AnomalyShield.
    hidden_dim : int
        Size of the LSTM hidden state.
    n_layers : int
        Number of stacked LSTM layers.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    threshold_percentile : float
        Percentile of training reconstruction errors used as the anomaly
        threshold.
    window_size : int
        Sliding window size applied when input is 2-D.  Ignored when input
        is already 3-D.
    batch_size : int
        Mini-batch size for training.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "LSTMAutoencoder",
        hidden_dim: int = 32,
        n_layers: int = 1,
        epochs: int = 50,
        lr: float = 1e-3,
        threshold_percentile: float = 95,
        window_size: int = 30,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            threshold_percentile=threshold_percentile,
            window_size=window_size,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.window_size = window_size
        self.batch_size = batch_size
        self.random_state = random_state

        self._model: LSTMAutoencoder | None = None
        self._threshold: float | None = None
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_windows(self, X: np.ndarray) -> np.ndarray:
        """Convert 2-D input to 3-D sliding windows; pass 3-D through."""
        if X.ndim == 3:
            return X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        if n_samples < self.window_size:
            raise ValueError(
                f"Input has {n_samples} samples, but window_size is "
                f"{self.window_size}. Provide more data or reduce window_size."
            )
        n_windows = n_samples - self.window_size + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (self.window_size, n_features)
        )
        return windows.reshape(n_windows, self.window_size, n_features)

    def _numpy_to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=self._device)

    def _reconstruction_errors(self, X_windows: np.ndarray) -> np.ndarray:
        """Compute per-window mean reconstruction error."""
        self._model.eval()  # type: ignore[union-attr]
        tensor = self._numpy_to_tensor(X_windows)
        with torch.no_grad():
            recon = self._model(tensor)  # type: ignore[misc]
        # MSE per window: mean over (seq_len, features)
        errors = ((tensor - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
        return errors

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"Detector '{self.name}' has not been fitted. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        """Train the LSTM autoencoder on the provided data.

        Parameters
        ----------
        X : np.ndarray
            Training data.  Accepts either a 2-D array of shape
            ``(n_samples, n_features)`` — which is internally converted to
            sliding windows — or a pre-windowed 3-D array of shape
            ``(n_windows, window_size, n_features)``.

        Returns
        -------
        AutoencoderDetector
            The fitted detector (self).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        windows = self._to_windows(X)
        input_dim = windows.shape[2]

        self._model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self._device)

        dataset = TensorDataset(self._numpy_to_tensor(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstruction = self._model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

        # Set anomaly threshold based on training reconstruction errors
        train_errors = self._reconstruction_errors(windows)
        self._threshold = float(np.percentile(train_errors, self.threshold_percentile))

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray
            Data of shape ``(n_samples, n_features)`` or
            ``(n_windows, window_size, n_features)``.

        Returns
        -------
        np.ndarray
            Labels: -1 for anomaly, 1 for normal.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        self._check_is_fitted()
        errors = self._reconstruction_errors(self._to_windows(X))
        return np.where(errors > self._threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous).

        Returns the mean reconstruction error per window.

        Parameters
        ----------
        X : np.ndarray
            Data of shape ``(n_samples, n_features)`` or
            ``(n_windows, window_size, n_features)``.

        Returns
        -------
        np.ndarray
            Anomaly scores where higher values indicate more anomalous samples.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        self._check_is_fitted()
        return self._reconstruction_errors(self._to_windows(X))
