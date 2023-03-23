import yaml
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model_architecture import MLPModel, MLPDataset
import tqdm
import matplotlib.pyplot as plt
import read_data
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(X, y, hyperparams:dict):
    model = MLPModel(num_features=X.shape[1], num_classes=hyperparams['num_classes'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss() # one-hot encoding taken care of by pytorch
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    train_generator = DataLoader(MLPDataset(X_train, y_train), batch_size=hyperparams['batch_size'])
    val_generator = DataLoader(MLPDataset(X_val, y_val), batch_size=hyperparams['batch_size'])
    
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, hyperparams['epochs'] + 1):
        print(f'Epoch {epoch}')
        
        batch_train_loss_history = []
        for batch_X, batch_y in tqdm(train_generator):
            optimizer.zero_grad()
            model.train()
            
            y_p = model(batch_X)
            loss = criterion(y_p, batch_y)

            loss.backward()
            optimizer.step()
            batch_train_loss_history.append(loss.item())
        
        batch_val_loss_history = []
        for batch_X, batch_y in tqdm(val_generator):
            model.eval()
            with torch.no_grad():
                y_p = model(X)
            
            loss = criterion(y_p, y)
            batch_val_loss_history.append(loss.item())
            
        # Append average loss across batches
        train_loss_history.append(sum(batch_train_loss_history) / len(batch_train_loss_history))
        val_loss_history.append(sum(batch_val_loss_history) / len(batch_val_loss_history))
        
    plt.figure()
    plt.title('Loss curve')
    plt.plot(range(hyperparams['epochs']), train_loss_history, label='train loss')
    plt.plot(range(hyperparams['epochs']), val_loss_history, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    with open('mlp_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)
    df = read_data.concat()
    y = pd.factorize(df.pop('classification'))[0]
    
    training_loop(df.to_numpy(), y, hyperparams)
    # training_loop(hyperparams)