import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None


class Autoencoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dims):
        if torch is None or nn is None or optim is None:
            raise ImportError(
                "torch is required for AutoencoderTransformer. Please install it using 'pip install your_package_name[torch]'."
            )
        super(Autoencoder, self).__init__()
        # Create the encoder
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Create the decoder
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoder_dims,
        decoder_dims,
        learning_rate=0.001,
        num_epochs=100,
        batch_size=32,
    ):
        self.input_dim = encoder_dims
        self.hidden_dim = decoder_dims
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.autoencoder = Autoencoder(encoder_dims, decoder_dims)

    def fit(self, X, y=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)

        # Normalize the functional connectivity data
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)

        data_loader = DataLoader(
            TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
            batch_size=self.batch_size,
        )

        for epoch in range(self.num_epochs):
            for data in data_loader:
                input_data = data[0].to(device)
                input_data = input_data.reshape(-1, self.input_dim)
                output = self.autoencoder(input_data)

                loss = criterion(output, input_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def transform(self, X, y=None):
        self.autoencoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return self.autoencoder.encoder(X_tensor).cpu().detach().numpy()


if __name__ == "__main__":
    fc_data = np.random.randn(
        100, 400
    )  # 400 is the example input dimension, adjust it to match your data
    input_dim = 400
    hidden_dim = 128
    autoencoder_transformer = AutoencoderTransformer(
        input_dim=input_dim, hidden_dim=hidden_dim
    )
    autoencoder_transformer.fit(fc_data)
    fc_data_transformed = autoencoder_transformer.transform(fc_data)
