import torch
import torch.nn as nn
import os

from config import  MODEL_PATH


def train(model, model_name, train_loader, val_loader, device, num_epochs, learning_rate=0.001):
    """
    Model training
    """
    print(f"{model_name} modell tanítása...")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward és optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Legjobb modell mentése
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.pth"))
            print(f"Modell mentve: {model_name}.pth")
    
    # Betöltjük a legjobb modellt
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name}.pth")))
    return model