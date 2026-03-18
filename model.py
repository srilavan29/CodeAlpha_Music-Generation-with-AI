import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, hidden_dim=512, n_layers=3):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, n_vocab)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x) # (batch, seq, embed)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
