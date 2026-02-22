"""
Model Handler - Wraps TensorFlow/Keras LSTM model for gesture prediction.
"""

import numpy as np


class GestureModel:
    """Handles loading and inference for the LSTM gesture recognition model."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self._build_model()

    def _build_model(self):
        """Build the LSTM architecture matching the trained notebook model."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense

            model = Sequential([
                LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 126)),
                LSTM(128, return_sequences=True, activation="relu"),
                LSTM(64, return_sequences=False, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(6, activation="softmax"),
            ])
            model.compile(
                optimizer="Adam",
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy"],
            )
            self.model = model
        except Exception as e:
            raise RuntimeError(f"Failed to build model architecture: {e}")

    def load(self, path: str = "action.h5"):
        """Load weights from .h5 file."""
        if self.model is None:
            self._build_model()
        self.model.load_weights(path)
        self.model_path = path

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Run inference on a sequence.

        Args:
            sequence: numpy array of shape (1, 30, 126)

        Returns:
            numpy array of shape (1, 6) with softmax probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.predict(sequence, verbose=0)

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Run batch inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.predict(sequences, verbose=0)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.model_path is not None

    def summary(self) -> str:
        if self.model:
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
            return stream.getvalue()
        return "Model not built"
