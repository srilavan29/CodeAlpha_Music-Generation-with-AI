import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from model import MusicLSTM

def train():
    # Load data
    network_input = np.load('processed_data/network_input.npy')
    network_output = np.load('processed_data/network_output.npy')
    
    with open('processed_data/mapping.pkl', 'rb') as f:
        note_to_int = pickle.load(f)
    
    n_vocab = len(note_to_int)
    
    # Convert to torch tensors
    inputs = torch.from_numpy(network_input).long()
    targets = torch.from_numpy(network_output).long()
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MusicLSTM(n_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    epochs = 10 # Reduced for demo
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1}/{epochs} Complete, Avg Loss: {total_loss/len(loader):.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_state_dict() if hasattr(model, 'state_state_dict') else model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "models/final_model.pth")
    print("Training complete. Model saved in models/")

if __name__ == '__main__':
    train()
