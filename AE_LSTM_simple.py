import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class PhaseFieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, dropout=0):
        super(PhaseFieldLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 16)
        #self.init_weights()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        #last_timestep = self.layer_norm(last_timestep)
        last_timestep = self.dropout(last_timestep)
        prediction = self.fc(last_timestep)
        return prediction
    
    def init_weights(self):
        """Initialize weights to improve training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class RollingWindowPredictor:
    def __init__(self, input_size, hidden_size=100, num_layers=2, learning_rate=1e-3, dropout=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhaseFieldLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,weight_decay=1e-6)
        self.training_losses = []
        self.validation_losses = []
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            cooldown=0,
            threshold=1e-3
        )
    
    def prepare_sequences(self, data, window_size=60, phase_idx=None):
        """
        Prepare sequences for training with configurable window size and optional phase selection
        
        Args:
        - data: numpy array of shape (num_simulations, timesteps, features)
        - window_size: number of timesteps to use as input
        - phase_idx: If specified, selects data for a single phase (0-4)
        
        Returns:
        - X: input sequences
        - y: target sequences
        """
        

        if phase_idx is not None:
            # Extract data for the specified phase
            X_Y_size = data.shape[2] // 5  # Size of X*Y for one phase
            start_idx = phase_idx * X_Y_size
            end_idx = (phase_idx + 1) * X_Y_size
            data = data[:, :, start_idx:end_idx]

        X, y = [], []
        for simulation in data:
            for i in range(len(simulation) - window_size - 1):
                X.append(simulation[i:i+window_size])
                y.append(simulation[i+window_size])
        
        X = np.array(X)
        y = np.array(y)

        indicies = np.random.permutation(len(X))

        X_shuffled = X[indicies]
        y_shuffled = y[indicies]
        
        return torch.FloatTensor(X_shuffled), torch.FloatTensor(y_shuffled)

    def train(self, data, window_size=60, epochs=100, batch_size=32, learning_rate=1e-5, test_split=0.1, phase_idx=None):
        """
        Train the model on either all phases or a single phase
        
        Args:
        - phase_idx: If specified (0-4), trains on single phase data
        """
        # Calculate input size based on data and phase selection
        if phase_idx is not None:
            input_size = data.shape[2] // 5  # Single phase size
        else:
            input_size = data.shape[2]  # Full size for all phases
            
        # Create new model with correct input size
        self.model = PhaseFieldLSTM(input_size, self.model.hidden_size, 
                                  self.model.num_layers).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=50,
            cooldown=50,
            threshold=1e-4
        )
        
        # Prepare sequences with specified window size and phase selection
        X, y = self.prepare_sequences(data, window_size, phase_idx)
        
        # Perform train/test split
        train_size = int(len(X) * (1 - test_split))
        print(f"Training Size is {train_size}")
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    predictions = self.model(batch_x)
                    loss = self.criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            val_loss /= len(test_dataloader)
            
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)
            self.scheduler.step(train_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {self.scheduler.get_last_lr()[0]}')

    def predict_rolling_window(self, initial_data, window_size, num_predictions, phase_idx=None):
        """
        Make predictions using either all phases or a single phase
        
        Args:
        - phase_idx: If specified (0-4), predicts single phase evolution
        """
        self.model.eval()
        
        if phase_idx is not None:
            # Extract data for the specified phase
            X_Y_size = initial_data.shape[1] // 5
            start_idx = phase_idx * X_Y_size
            end_idx = (phase_idx + 1) * X_Y_size
            initial_data = initial_data[:, start_idx:end_idx]

        current_window = initial_data
        predicted_sequence = []

        with torch.no_grad():
            for _ in range(num_predictions):
                input_tensor = torch.FloatTensor(current_window[-window_size:])
                
                if input_tensor.dim() == 2:
                    input_tensor = input_tensor.unsqueeze(0)
                
                input_tensor = input_tensor.to(self.device)
                prediction = self.model(input_tensor).cpu().numpy()
                predicted_sequence.append(prediction)

                current_window = np.concatenate([current_window, prediction])

        return np.array(predicted_sequence).squeeze()

    def visualize_results(self, true_data, predicted_data, phase_idx=None):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        plt.subplot(1, 2, 2)
        num_features = min(4, true_data.shape[1])
        phase_label = f" (Phase {phase_idx})" if phase_idx is not None else ""
        
        for i in range(num_features):
            plt.plot(true_data[:, i], label=f'True Feature {i}{phase_label}', linestyle='--')
            plt.plot(predicted_data[:, i], label=f'Predicted Feature {i}{phase_label}', linestyle='-')
        
        plt.title(f'True vs Predicted Timesteps{phase_label}')
        plt.xlabel('Timestep')
        plt.ylabel('Feature Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_with_context(self, context_data, true_data, predicted_data, phase_idx=None):
        
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.training_losses, label='Training Loss',linewidth=3)
        plt.plot(self.validation_losses, label='Validation Loss',linewidth=3)
        plt.title('Training vs Validation Loss',fontsize=24)
        plt.xlabel('Epochs',fontsize=24)
        plt.tick_params(axis='x', labelsize=20)
        plt.ylabel('Loss',fontsize=24)
        plt.tick_params(axis='y', labelsize=20)
        #plt.yscale('log')
        plt.ylim(1e-2,1e+1)
        plt.legend()

        plt.subplot(1, 2, 2)

        num_features = min(4, true_data.shape[1])
        phase_label = f" (Phase {phase_idx})" if phase_idx is not None else ""
        context_length = len(context_data)

        for i in range(num_features):
            #i_star = 4*i
            i_star = i
            # Plot context data
            plt.plot(range(1, context_length + 1), 
                    context_data[:, i_star], 
                    label=f'Feature {i} Context{phase_label}',
                    color=f'C{i}', linewidth=4)
            '''
            # Plot true and predicted data after context
            plt.plot(range(context_length, context_length + len(true_data) + 1),
                    np.concatenate([[context_data[-1, i_star]], true_data[:, i_star]]),
                    label=f'True Feature {i}{phase_label}',
                    linestyle='--', color=f'C{i}', linewidth=4)
            
            plt.plot(range(context_length, context_length + len(predicted_data) + 1),
                    np.concatenate([[context_data[-1, i_star]], predicted_data[:, i_star]]),
                    label=f'Predicted Feature {i}{phase_label}',
                    linestyle=':', color=f'C{i}', linewidth=4)
        
        # Add vertical line to mark end of context
        plt.axvline(x=context_length, color='gray', linestyle='-', alpha=0.3)'
        '''
        #plt.text(context_length - 1, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')
        
        plt.title(f'LSTM Predictions with Context{phase_label}')
        plt.xlabel('Timestep', fontsize=48)
        plt.tick_params(axis='x', labelsize=40)
        plt.xticks([0, 20, 40, 60, 80, 100])
        plt.ylabel('Feature Value',fontsize=48)
        plt.tick_params(axis='y', labelsize=48)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def loadAEdata(filepath, t=101, phases=5):
    raw_data = np.load(filepath)
    print(raw_data.shape)
    frames_per_sim = t*phases
    num_sim = int(raw_data.shape[0]/frames_per_sim)
    X = raw_data.shape[1]
    Y= raw_data.shape[2]
    temp1 = raw_data.reshape(num_sim, frames_per_sim, X, Y)
    temp2 = temp1.reshape(num_sim, phases, t, X, Y)
    temp3 = np.transpose(temp2, (0,2,1,3,4))
    arr = temp3.reshape(num_sim, t, phases*X*Y)
    print(arr.shape)
    return arr

if __name__ == "__main__":
    # Example usage
    AEfile = 'ost_AE.npy'
    simulations = loadAEdata(AEfile)
    
    input_window = 100
    n_predictions = 100-input_window

    train = True
    
    # Example for single phase prediction (phase 0)
    phase_idx = 0  # Change this to predict different phases (0-4)
    
    # Calculate input size based on the data shape and phase selection
    input_size = simulations.shape[2] // 5 if phase_idx is not None else simulations.shape[2]

    
    
    predictor = RollingWindowPredictor(
        input_size=input_window*input_size,
        hidden_size=8*16,
        num_layers=2,
        dropout=0.2
    )
    initial_sequence = simulations[121, :input_window]
    true_future = simulations[121, input_window:input_window+n_predictions, 16*phase_idx:16*(phase_idx+1)]
    predicted_future = simulations[121, input_window:input_window+n_predictions, 16*phase_idx:16*(phase_idx+1)]
    predictor.plot_with_context(initial_sequence[:,16*phase_idx:16*(phase_idx+1)],true_future,predicted_future,phase_idx=phase_idx)

    if train:
        t0 = time.time()
        predictor.train(
            simulations.copy(),
            window_size=input_window,
            epochs=1000,
            batch_size=512,
            learning_rate=1e-3,
            phase_idx=phase_idx,
            test_split=0.2
        )
        t1 = time.time()
    

    # Make predictions for the single phase # Used 55 previously, now 7
    initial_sequence = simulations[7, :input_window]
    true_future = simulations[7, input_window:input_window+n_predictions, 16*phase_idx:16*(phase_idx+1)]
    t2 =  time.time()
    predicted_future = predictor.predict_rolling_window(
        initial_data=initial_sequence,
        window_size=input_window,
        num_predictions=n_predictions,
        phase_idx=phase_idx
    )
    t3 = time.time()

    input(f"Time to train model is {t1-t0}s. Time to predict sequence is {t3-t2}s")

    true_img = true_future.reshape((true_future.shape[0],4,4))
    pred_img = predicted_future.reshape((predicted_future.shape[0],4,4))

    vmin = min(true_img.min(), pred_img.min())
    vmax = max(true_img.max(), pred_img.max())

    fig, axes = plt.subplots(2, 10, figsize=(12, 3)) 

    # Loop through the first row (original images)
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(true_img[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Frame {input_window+i}")
        ax.axis('off')  # Hide axes

    # Loop through the second row (comparison images)
    for i, ax in enumerate(axes[1]):
        im = ax.imshow(pred_img[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')  # Hide axes

    axes[0, 0].set_ylabel("True", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Predicted", fontsize=12, fontweight='bold')

    # Add a single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

    predictor.plot_with_context(initial_sequence[:,16*phase_idx:16*(phase_idx+1)],true_future,predicted_future,phase_idx=phase_idx)

    input("Press Enter to Save or quit to quit")
    
    np.save('predicted_images.npy',pred_img)

    # Visualize results for the single phase
    #predictor.visualize_results(true_future, predicted_future, phase_idx=phase_idx)
    
    