import numpy as np
import pandas as pd
#from sklearn.svm import SVR
from cuml.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import seaborn as sns
import joblib
#from numba import cuda

def prepare_data_single_target(data, target_field=4, target_pca=0):
    """
    Prepare data for predicting a single PCA component of one field using all components of other fields.
    
    Parameters:
    data: numpy array of shape (n_simulations, n_timesteps, n_fields, n_pca_components)
    target_field: index of the field to predict (default 4 for eta4)
    target_pca: index of the PCA component to predict
    
    Returns:
    X: Features array containing all PCA components of other fields
    y: Target array containing single PCA component of target field
    """
    n_sims, n_times, n_fields, n_pca = data.shape
    
    # Reshape to 2D array where each row contains all PCA components for all fields
    # for a specific simulation and timestep
    X = data.reshape(-1, n_fields * n_pca)
    
    # Extract target value (single PCA component of target field)
    y = X[:, target_field * n_pca + target_pca]
    
    # Remove all PCA components of target field from features
    mask = np.ones(X.shape[1], dtype=bool)
    mask[target_field * n_pca:(target_field + 1) * n_pca] = False
    X = X[:, mask]
    
    return X, y

def prepare_data_multi_target(data, target_fields=[3, 4], target_pca=0):
    """
    Prepare data for predicting multiple PCA components of multiple fields using components of other fields.
    
    Parameters:
    data: numpy array of shape (n_simulations, n_timesteps, n_fields, n_pca_components)
    target_fields: list of indices of the fields to predict
    target_pca: index of the PCA component to predict
    
    Returns:
    X: Features array containing all PCA components of non-target fields
    y_list: List of target arrays containing PCA components of target fields
    """
    n_sims, n_times, n_fields, n_pca = data.shape
    
    # Reshape to 2D array where each row contains all PCA components for all fields
    X = data.reshape(-1, n_fields * n_pca)
    
    # Extract target values (PCA components of target fields)
    y_list = [X[:, field * n_pca + target_pca] for field in target_fields]
    
    # Remove all PCA components of target fields from features
    mask = np.ones(X.shape[1], dtype=bool)
    for field in target_fields:
        mask[field * n_pca:(field + 1) * n_pca] = False
    X = X[:, mask]
    
    return X, y_list

def train_svr_model(X, y, kernel='rbf'):
    """
    Train SVR model for single target prediction.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        
        'C': [1000], 
        'epsilon': [0.1],
        'gamma': ['scale'],
        'kernel': ['rbf'] 
    }
    """
        'C': [0.1, 1, 10, 100], #500
        'epsilon': [0.01, 0.1, 0.2], #0.01, 0.1
        'gamma': ['scale', 'auto', 0.1, 0.01] #'scale','auto',0.01
    """
    
    # Initialize SVR
    print("Initialize SVR")
    svr = SVR()
    
    # Perform grid search
    print("Perform Grid Search")
    grid_search = GridSearchCV(svr, param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1) # cv should be 5
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    print("Make Predictions")
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    print("Calculate Metrics")
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return best_model, scaler, (X_test, y_test, y_pred), r2, rmse, grid_search.best_params_

def train_multi_svr_model(X, y_list, kernel='rbf'):
    """
    Train multiple SVR models for multiple target predictions.
    """
    models = []
    scalers = []
    metrics = []
    results = []
    
    for target_idx, y in enumerate(y_list):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grid
        param_grid = {
            'C': [10,100],
            'epsilon': [0.1,0.01],
            'gamma': ['auto','scale']
        }
        
        # Initialize SVR
        print(f"\nTraining model for target field {target_idx}")
        svr = SVR(kernel=kernel)
        
        # Perform grid search
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        models.append(best_model)
        scalers.append(scaler)
        metrics.append({'r2': r2, 'rmse': rmse, 'best_params': grid_search.best_params_})
        results.append((X_test, y_test, y_pred))
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    return models, scalers, results, metrics

def analyze_temporal_correlations(data):
    """
    Analyze correlations between first PCA component across time for each simulation.
    Returns both mean correlations and correlation distributions.
    """
    n_sims, n_times, n_fields, n_pca = data.shape
    correlations = np.zeros((n_sims, n_fields, n_fields))
    
    for sim in range(n_sims):
        for field1 in range(n_fields):
            for field2 in range(n_fields):
                # Calculate correlation between time series of first PCA component
                corr, _ = pearsonr(data[sim, :, field1, 0], data[sim, :, field2, 0])
                correlations[sim, field1, field2] = corr
    
    mean_correlations = np.mean(correlations, axis=0)
    
    return mean_correlations, correlations

def plot_correlation_analysis(correlations, field_names):
    """
    Create comprehensive correlation analysis plots.
    """
    mean_correlations, all_correlations = correlations
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Heatmap of mean correlations
    sns.heatmap(mean_correlations, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                vmin=-1, 
                vmax=1,
                xticklabels=field_names,
                yticklabels=field_names,
                ax=ax1)
    ax1.set_title('Mean Temporal Correlations')
    
    # Plot 2: Box plot of correlation distributions
    correlation_data = []
    for i in range(len(field_names)):
        for j in range(i+1, len(field_names)):
            pairs = pd.DataFrame({
                'Fields': f'{field_names[i]} vs {field_names[j]}',
                'Correlation': all_correlations[:, i, j]
            })
            correlation_data.append(pairs)
    
    if correlation_data:
        correlation_df = pd.concat(correlation_data, ignore_index=True)
        sns.boxplot(x='Fields', y='Correlation', data=correlation_df, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title('Distribution of Correlations Across Simulations')
    
    plt.tight_layout()
    return fig

def plot_prediction_results(results, title="Actual vs Predicted Values"):
    """
    Plot actual vs predicted values with additional analysis.
    """
    X_test, y_test, y_pred = results
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot of predicted vs actual values
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual Values')
    
    # Residual plot
    residuals = y_pred - y_test
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def run_complete_analysis(data, target_field=3, field_names=None):
    """
    Run complete analysis including correlations and SVR prediction.
    """
    if field_names is None:
        field_names = [f'Field_{i}' for i in range(data.shape[2])]
    
    # Analyze correlations
    
    correlations = analyze_temporal_correlations(data)
    '''
    try:
        correlation_fig = plot_correlation_analysis(correlations, field_names)
    except Exception as e:
        print(f"Warning: Could not create correlation plot: {str(e)}")
        correlation_fig = None
    correlation_fig.show
    '''
    
    # Train models for each PCA component
    results = []
    pca_range = range(data.shape[3])
    #pca_range = range(1)
    for pca_idx in pca_range:
        print(f"\nPreparing data for PCA component {pca_idx + 1}")
        X, y = prepare_data_single_target(data, target_field=target_field, target_pca=pca_idx)
        print(f"\nTraining model for PCA component {pca_idx + 1}")
        model, scaler, pred_results, r2, rmse, best_params = train_svr_model(X, y)
        
        results.append({
            'pca_component': pca_idx,
            'model': model,
            'scaler': scaler,
            'r2': r2,
            'rmse': rmse,
            'best_params': best_params,
            'prediction_results': pred_results
        })
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Best parameters: {best_params}")
        '''
        try:
            plot_prediction_results(pred_results, 
                                  title=f'Predictions for PCA Component {pca_idx + 1}')
        except Exception as e:
            print(f"Warning: Could not create prediction plot for component {pca_idx + 1}: {str(e)}")
        '''
    
    return results, correlations

def run_complete_multi_analysis(data, target_fields=[3, 4], field_names=None):
    """
    Run complete analysis for multiple target fields.
    """
    if field_names is None:
        field_names = [f'Field_{i}' for i in range(data.shape[2])]
    
    # Train models for each PCA component
    all_results = []
    for pca_idx in range(data.shape[3]):
        print(f"\nPreparing data for PCA component {pca_idx + 1}")
        X, y_list = prepare_data_multi_target(data, target_fields=target_fields, target_pca=pca_idx)
        
        print(f"\nTraining models for PCA component {pca_idx + 1}")
        models, scalers, pred_results, metrics = train_multi_svr_model(X, y_list)
        
        all_results.append({
            'pca_component': pca_idx,
            'models': models,
            'scalers': scalers,
            'metrics': metrics,
            'prediction_results': pred_results
        })
        
        # Plot results for each target field
        '''
        for i, results in enumerate(pred_results):
            try:
                plot_prediction_results(results, 
                                     title=f'Predictions for Field {target_fields[i]}, PCA Component {pca_idx + 1}')
            except Exception as e:
                print(f"Warning: Could not create prediction plot for field {target_fields[i]}, component {pca_idx + 1}: {str(e)}")
        '''
    
    return all_results

def reshape_grouped_data(data):
    """
    Reshape data from (500, 100, 5) where each series contains 5 grouped fields
    to (100, 100, 5, 5) format for (simulations, timesteps, fields, pca_components)
    
    Parameters:
    data: numpy array of shape (500, 100, 5) containing grouped series
    
    Returns:
    reshaped_data: numpy array of shape (100, 100, 5, 5)
    """
    # Verify input shape
    n_images, n_x, n_y = data.shape
    n_pca = n_x * n_y

    n_timesteps = 101
    n_phases = 5

    assert n_images % (n_timesteps*n_phases) == 0, "Number of images must be divisible by 505"
    
    n_series = n_images // n_timesteps
    n_simulations = n_series // n_phases
    
    # Reshape data
    # First reshape to group related fields together
    first_reshape = data.reshape(n_images, n_pca)
    second_reshape = first_reshape.reshape(n_series, n_timesteps, n_pca)
    intermediate = second_reshape.reshape(n_simulations, n_phases, n_timesteps, n_pca)
    # Transpose to get desired order of dimensions
    reshaped_data = np.transpose(intermediate, (0, 2, 1, 3))

    
    return reshaped_data

def verify_reshape(original_data, reshaped_data):
    """
    Verify that the reshaping preserved the data correctly
    
    Parameters:
    original_data: numpy array of shape (10100, 4, 4)
    reshaped_data: numpy array of shape (20, 100, 5, 16)
    
    Returns:
    bool: True if reshaping was successful
    """
    # Get original shape parameters
    print(original_data.shape)
    print(reshaped_data.shape)
    n_simulations, n_timesteps, n_phases, n_pca = reshaped_data.shape
    

    # Check a few random indices
    np.random.seed(43)
    for _ in range(5):
        # Random simulation and timestep
        sim_idx = np.random.randint(0, n_simulations-1)
        time_idx = np.random.randint(0, n_timesteps-1)
        field_idx = np.random.randint(0, 4)

        x_idx = np.random.randint(0,3)
        y_idx = np.random.randint(0,3)
        
        # Calculate corresponding index in original data

        original_idx = sim_idx * n_phases * n_timesteps + field_idx * n_timesteps + time_idx
        pca_idx = x_idx * 4 + y_idx 
        
        # Compare values
        original_value = original_data[original_idx, x_idx, y_idx]
        reshaped_value = reshaped_data[sim_idx, time_idx, field_idx, pca_idx]

        if not np.allclose(original_value, reshaped_value):
            return False
    
    return True


def process_and_analyze_data(data):
    """
    Process the data and print verification information
    
    Parameters:
    data: numpy array of shape (101000, 4, 4)
    
    Returns:
    reshaped_data: numpy array of shape (200, 101, 5, 16)
    """
    # Reshape the data
    reshaped_data = reshape_grouped_data(data)
    
    # Verify the reshape
    is_valid = verify_reshape(data, reshaped_data)
    
    # Print information about the reshape
    print(f"Original shape: {data.shape}")
    print(f"Reshaped shape: {reshaped_data.shape}")
    print(f"Reshape validation {'successful' if is_valid else 'failed'}")
    
    # Print example of how the data is organized
    print("\nExample data organization:")
    simulation = 1
    timestep = 1
    print(f"Simulation {simulation}, Timestep {timestep}:")
    for field in range(5):
        print(f"\nField {field} Encoded Values:", reshaped_data[simulation, timestep, field])
    print("\n")

    #input(f'Original Shape: {data.shape}. New Shape: {reshaped_data.shape}. Quit now...')
    
    return reshaped_data

def predictSinglePhase(arr, model_list, target):
    #input(arr.shape)
    num_pca = arr.shape[3]
    results = np.empty((arr.shape[0]*arr.shape[1],1,arr.shape[3]))
    for pca in range(num_pca):
        scaler = StandardScaler()
        X, y = prepare_data_single_target(arr, target, pca)
        X_scaled = scaler.fit_transform(X)
        model = model_list[pca]
        y_pred = model.predict(X_scaled)
        results[:,0,pca] = y_pred
        print(f"Predicted pixel {pca+1} ({arr.shape[0]*arr.shape[1]} frames)")
    return results

def predict_multiple_phases(arr, model_lists, target_fields=[3, 4]):
    """
    Predict multiple phases using trained models.
    """
    num_pca = arr.shape[3]
    num_targets = len(target_fields)
    results = np.empty((arr.shape[0]*arr.shape[1], num_targets, arr.shape[3]))
    
    for pca in range(num_pca):
        scaler = StandardScaler()
        X, _ = prepare_data_multi_target(arr, target_fields=target_fields, target_pca=pca)
        X_scaled = scaler.fit_transform(X)
        
        for target_idx in range(num_targets):
            model = model_lists[pca][target_idx]
            y_pred = model.predict(X_scaled)
            results[:, target_idx, pca] = y_pred
            print(f"Predicted pixel {pca+1} for phase {target_fields[target_idx]} ({arr.shape[0]*arr.shape[1]} frames)")
    
    return results


def savemodels(model_list, save_folder):

    n_model = len(model_list)
    for i in range(n_model):
        save_model = model_list[i]
        save_path = f"{save_folder}/SVR_{i}.pkl"
        #save_file = open(save_path, "w")
        #save_file.write("")
        #save_file.close
        joblib.dump(save_model, save_path)
    return 0

def loadmodels(n_model, save_folder):

    model_list = []
    for i in range(n_model):
        load_path = f"{save_folder}/SVR_{i}.pkl"
        load_model = joblib.load(load_path)
        model_list.append(load_model)
    return model_list

def save_multi_models(model_lists, save_folder, prefix="SVR"):
    """
    Save multiple SVR models to files.
    """
    n_components = len(model_lists)
    n_targets = len(model_lists[0])
    
    for i in range(n_components):
        for j in range(n_targets):
            save_path = f"{save_folder}/{prefix}_{i}_{j}.pkl"
            joblib.dump(model_lists[i][j], save_path)
    return 0

def load_multi_models(n_components, n_targets, save_folder, prefix="SVR"):
    """
    Load multiple SVR models from files.
    """
    model_lists = []
    for i in range(n_components):
        models = []
        for j in range(n_targets):
            load_path = f"{save_folder}/{prefix}_{i}_{j}.pkl"
            model = joblib.load(load_path)
            models.append(model)
        model_lists.append(models)
    return model_lists

def visualize_phase_predictions(actual_data, predictions, target_fields=4, timesteps=[0, 25, 50, 75, 100], figsize=(15, 10)):
    """
    Visualize phase predictions with actual vs predicted comparisons at specific timesteps.
    Works with both single and multiple phase predictions.
    
    Parameters:
    actual_data: numpy array of shape (n_simulations, n_timesteps, n_fields, n_pca_components)
    predictions: For single phase: array of shape (n_total_frames, 1, n_pca_components)
                For multiple phases: array of shape (n_total_frames, n_targets, n_pca_components)
    target_fields: int or list of ints indicating which fields to predict
    timesteps: list of timesteps to visualize
    """
    n_sims = actual_data.shape[0]
    n_times = actual_data.shape[1]
    
    # Convert target_fields to list if it's a single integer
    if isinstance(target_fields, int):
        target_fields = [target_fields]
    n_targets = len(target_fields)
    
    # Reshape predictions back to match actual data format
    if predictions.shape[1] == 1:  # Single phase prediction
        pred_reshaped = [predictions[:, 0].reshape(n_sims, n_times, -1)]
    else:  # Multiple phase prediction
        pred_reshaped = [predictions[:, i].reshape(n_sims, n_times, -1) for i in range(n_targets)]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    n_cols = len(timesteps)
    n_rows = 2 * n_targets  # Actual and predicted for each target
    gs = GridSpec(n_rows, n_cols + 1)  # +1 for colorbar
    
    # Plot phases at selected timesteps
    vmin = min(actual_data.min(), min(p.min() for p in pred_reshaped))
    vmax = max(actual_data.max(), max(p.max() for p in pred_reshaped))
    
    for t_idx, timestep in enumerate(timesteps):
        for target_idx, target_field in enumerate(target_fields):
            # Plot actual phase
            ax1 = fig.add_subplot(gs[target_idx * 2, t_idx])
            im1 = ax1.imshow(actual_data[0, timestep, target_field].reshape(4, 4), 
                           vmin=vmin, vmax=vmax, cmap='viridis')
            ax1.set_title(f'Actual η{target_field}\nt={timestep}')
            ax1.axis('off')
            
            # Plot predicted phase
            ax2 = fig.add_subplot(gs[target_idx * 2 + 1, t_idx])
            im2 = ax2.imshow(pred_reshaped[target_idx][0, timestep].reshape(4, 4), 
                           vmin=vmin, vmax=vmax, cmap='viridis')
            ax2.set_title(f'Predicted η{target_field}\nt={timestep}')
            ax2.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im1, cax=cbar_ax)
    
    plt.tight_layout()
    return fig

def plot_prediction_statistics(actual_data, predictions, target_fields=4, figsize=(15, 5)):
    """
    Plot statistical analysis of predictions including error distributions and correlations.
    Works with both single and multiple phase predictions.
    
    Parameters:
    actual_data: numpy array of shape (n_simulations, n_timesteps, n_fields, n_pca_components)
    predictions: For single phase: array of shape (n_total_frames, 1, n_pca_components)
                For multiple phases: array of shape (n_total_frames, n_targets, n_pca_components)
    target_fields: int or list of ints indicating which fields to predict
    """
    if isinstance(target_fields, int):
        target_fields = [target_fields]
    n_targets = len(target_fields)
    
    # Create figure based on number of targets
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_targets, height_ratios=[7, 1])
    
    # Reshape data for analysis
    for target_idx, target_field in enumerate(target_fields):
        # Get actual values
        actual_values = actual_data[:, :, target_field].reshape(-1)
        
        # Get predicted values (handle both single and multi-phase cases)
        if predictions.shape[1] == 1:  # Single phase
            predicted_values = predictions[:, 0].reshape(-1)
        else:  # Multiple phases
            predicted_values = predictions[:, target_idx].reshape(-1)
        
        
        # Recreate the same split with the same seed
        indices = np.arange(len(actual_values))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=42  # Same seed as before
        )

        # Create a mask for training and validation points
        is_train = np.zeros(len(actual_values), dtype=bool)
        is_train[train_indices] = True


        # Calculate metrics
        mse_train = mean_squared_error(actual_values[is_train], predicted_values[is_train])
        r2_train = r2_score(actual_values[is_train], predicted_values[is_train])

        mse_val = mean_squared_error(actual_values[~is_train], predicted_values[~is_train])
        r2_val = r2_score(actual_values[~is_train], predicted_values[~is_train])

        # Scatter plot with different colors
        plt.rcParams.update({'font.size': 14})
        ax1 = fig.add_subplot(gs[0, target_idx])
        #ax1.scatter(actual_values, predicted_values, alpha=0.2, s=2)
        ax1.scatter(actual_values[is_train], predicted_values[is_train],
                alpha=0.4, s=1, color='red', label=f'Training: MSE={mse_train:.4f}')
        ax1.scatter(actual_values[~is_train], predicted_values[~is_train],
                alpha=0.4, s=1, color='blue', label=f'Validation: MSE={mse_val:.4f}')
        ax1.plot([actual_values.min(), actual_values.max()],
                [actual_values.min(), actual_values.max()],
                color='gray', linestyle='--', linewidth=2)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Actual Values')
        ax1.set_xlim(-90, 120)
        ax1.set_xticks([-60,-30,0,30,60,90])
        ax1.set_ylabel('Predicted Values')
        ax1.set_ylim(-90, 120)
        ax1.set_yticks([-60,-30,0,30,60,90])
        ax1.legend(loc='upper left')
        if target_field!=0:
            ax1.set_title(f'η{target_field} Predictions')
        else:
            ax1.set_title(f'C Predictions')
        
        # Error distribution
        ax2 = fig.add_subplot(gs[1, target_idx])
        errors = predicted_values - actual_values
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title(f'η{target_field} Error Distribution\nμ: {errors.mean():.4f}, σ: {errors.std():.4f}')
    
    plt.tight_layout()
    return fig

def analyze_temporal_evolution(actual_data, predictions, target_fields=4, figsize=(15, 5)):
    """
    Analyze how prediction accuracy evolves over time.
    Works with both single and multiple phase predictions.
    
    Parameters:
    actual_data: numpy array of shape (n_simulations, n_timesteps, n_fields, n_pca_components)
    predictions: For single phase: array of shape (n_total_frames, 1, n_pca_components)
                For multiple phases: array of shape (n_total_frames, n_targets, n_pca_components)
    target_fields: int or list of ints indicating which fields to predict
    """
    if isinstance(target_fields, int):
        target_fields = [target_fields]
        
    n_sims = actual_data.shape[0]
    n_times = actual_data.shape[1]
    n_targets = len(target_fields)
    
    # Calculate MSE over time for each target
    time_mse = np.zeros((n_times, n_targets))
    time_r2 = np.zeros((n_times, n_targets))
    
    # Reshape predictions
    pred_reshaped = []
    if predictions.shape[1] == 1:  # Single phase
        pred_reshaped = [predictions[:, 0].reshape(n_sims, n_times, -1)]
    else:  # Multiple phases
        pred_reshaped = [predictions[:, i].reshape(n_sims, n_times, -1) for i in range(n_targets)]
    
    for t in range(1,n_times): #starting with 1
        for target_idx, target_field in enumerate(target_fields):
            actual = actual_data[:, t, target_field].reshape(-1)
            pred = pred_reshaped[target_idx][:, t].reshape(-1)
            time_mse[t, target_idx] = mean_squared_error(actual, pred)
            time_r2[t, target_idx] = r2_score(actual, pred)
    
    # Plot temporal evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for target_idx, target_field in enumerate(target_fields):
        ax1.plot(range(1,n_times), time_mse[1:, target_idx], 
                label=f'η{target_field}')
        ax2.plot(range(1,n_times), time_r2[1:, target_idx], 
                label=f'η{target_field}')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Prediction Error Over Time')
    ax1.legend()
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Prediction Accuracy Over Time')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def remove_t_0(full_data, predictions):
    full_data_cull = full_data[:,1:,:,:]
    predictions_reshape =  predictions.reshape((full_data.shape[0],full_data.shape[1],2,16))
    predictions_reshape_cull = predictions_reshape[:,1:,:,:]
    predictions_cull = predictions_reshape_cull.reshape((predictions_reshape_cull.shape[0]*predictions_reshape_cull.shape[1],predictions_reshape_cull.shape[2],predictions_reshape_cull.shape[3]))
    return(full_data_cull, predictions_cull)



microstructure = 'ost'
encoded_file = microstructure + '_AE.npy'
mode = 'single'
train = True


raw_data = np.load(encoded_file)
print(raw_data.shape)
data = process_and_analyze_data(raw_data)[:]

# Define field names
field_names = ['composition','eta1', 'eta2', 'eta3', 'eta4']

if mode == 'single':
    target_field = 0
    if train:
        results, correlations = run_complete_analysis(data, target_field=target_field, field_names=field_names)

        models = []
        for resultdict in results:
            models.append(resultdict.get('model'))
        savemodels(models,'SVR_models')

    full_data_raw = np.load(encoded_file)
    full_data = process_and_analyze_data(full_data_raw)
    model_list = loadmodels(16,'SVR_models')

    #predictions = predictSinglePhase(full_data,model_list)
    predictions = predictSinglePhase(data,model_list,target_field)
    np.save(f'pred_{field_names[target_field]}.npy',predictions.reshape(predictions.shape[0],4,4))

    # Visualize
    fig1 = visualize_phase_predictions(full_data, predictions, target_fields=target_field)
    fig2 = plot_prediction_statistics(full_data, predictions, target_fields=target_field)
    fig3 = analyze_temporal_evolution(full_data, predictions, target_fields=target_field)

    plt.show()




if mode == 'double':
    target_fields=[3,4]
    if train:
        results = run_complete_multi_analysis(data, target_fields=target_fields, field_names=field_names)

        # Save models
        model_lists = []
        for result in results:
            model_lists.append(result['models'])
        save_multi_models(model_lists, 'SVR_models_multi')

    # Prediction
    full_data_raw = np.load(encoded_file)
    full_data = process_and_analyze_data(full_data_raw)
    model_lists = load_multi_models(16, 2, 'SVR_models_multi')  # 16 PCA components, 2 target fields
    predictions = predict_multiple_phases(full_data, model_lists, target_fields=[3, 4])

    # Visualize predictions at specific timesteps
    fig1 = visualize_phase_predictions(full_data, predictions, target_fields=[3, 4])
    plt.savefig('phase_predictions.png')

    # Plot statistical analysis
    fig2 = plot_prediction_statistics(full_data, predictions, target_fields=[3, 4])
    plt.savefig('prediction_statistics.png')

    full_data, predictions = remove_t_0(full_data, predictions)

    # Analyze temporal evolution
    fig3 = analyze_temporal_evolution(full_data, predictions, target_fields=[3, 4])
    plt.savefig('temporal_evolution.png')

    plt.show()

    # Save predictions - reshape for each target field
    for i, field in enumerate(target_fields):
        np.save(f'pred_eta{field+1}.npy', predictions[:, i].reshape(-1, 4, 4))

input("Press Enter to Exit...")
