import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """
    A class representing a neural network model for predicting signal strength based on position.

    Attributes:
    - input_dim (int): The input dimension of the data.
    - layer_wid (list): A list defining the width of each layer.
    - nonlinearity (str): The type of nonlinearity to apply between layers.
    """
    def __init__(self, input_dim, layer_wid, nonlinearity):
        """
        Initializes the Net class by building a feed-forward neural network.

        Parameters:
        - input_dim (int): The number of input features.
        - layer_wid (list): List of integers defining the number of neurons in each layer.
        - nonlinearity (str): The nonlinearity function to use ('relu', 'sigmoid', etc.).

        Outputs:
        - Initializes the network with specified architecture and activation functions.
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        self.normalization = nn.LayerNorm(input_dim)  # Use LayerNorm for input normalization
        self.dropout = nn.Dropout(p=0.2)  # Add Dropout layer
        self.fc_layers = nn.ModuleList()

        # Create input layer
        self.fc_layers.append(nn.Linear(in_features=input_dim, out_features=layer_wid[0]))

        # Create hidden layers
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], out_features=layer_wid[i + 1]))

        # Apply Xavier Initialization
        self.initialize_weights()


        # LayerNorm per hidden-layer input (stable for batch size 1 / FL)
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(layer.in_features) for layer in self.fc_layers[:-1]])
        # Set the activation function
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
    
    def initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                torch.nn.init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:  # Initialize bias as zero
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for the neural network.

        Parameters:
        - x (Tensor): Input tensor with dimensions N x input_dim (N = batch size).

        Outputs:
        - (Tensor): Output tensor after passing through the network.
        """
        # Input normalization (LayerNorm is stable for small batches / FL)
        x = self.normalization(x)

        # Hidden layers
        for fc_layer, norm in zip(self.fc_layers[:-1], self.hidden_norms):
            # Normalize the input to each layer (replaces forward-created BatchNorm)
            x = norm(x)
            x = self.nonlinearity(fc_layer(x))
            x = self.dropout(x)

        # Output layer
        return self.fc_layers[-1](x)


    def get_layers(self):
        """
        Retrieves the input and output dimensions of all layers.

        Outputs:
        - (list): A list containing the input and output dimensions of all layers.
        """
        L = len(self.fc_layers)
        layers = (L + 1) * [0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i + 1] = self.fc_layers[i].out_features
        return layers

    def get_param(self):
        """
        Returns a flattened tensor of all parameters in the network.

        Outputs:
        - (Tensor): A concatenated tensor of all network parameters.
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P, a))
        return P

class Polynomial3(nn.Module):
    """
    A class representing a pathloss model for predicting signal strength based on position.

    Attributes:
    - gamma (float): A learnable parameter representing the path loss exponent.
    - theta (Parameter): A learnable parameter representing the jammer's position.
    - P0 (Parameter): The transmit power to be learned.
    """
    def __init__(self, gamma=2, theta0=None, P0_init=None):
        """
        Initializes the Polynomial3 class with the specified parameters.

        Parameters:
        - gamma (float): Path loss exponent (default is 2).
        - theta0 (list or None): Initial value for theta (position).
        - P0_init (float or None): Initial value for P0 (reference power at 1m).
            If None, defaults to 10 (Jaramillo's original).
            Recommended: estimate from data using estimate_P0_from_data().

        Outputs:
        - Initializes the polynomial model with learnable parameters.
        """
        super().__init__()
        self.theta = nn.Parameter(torch.zeros((2))) if theta0 is None else nn.Parameter(torch.tensor(theta0))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        
        # P0 initialization: use provided value or default to 10 (Jaramillo's original)
        if P0_init is None:
            P0_init = 10.0
        self.P0 = nn.Parameter(torch.tensor(float(P0_init), dtype=torch.float32))
    
    def forward(self, x):
        """
        Forward pass for the path-loss (physics) model.

        Uses only position columns (x_enu, y_enu) in the first 2 dims.

        Model:
            RSSI = P0 - 10*gamma*log10(d)

        Numerical stability:
            - Adds a small epsilon to distance
            - Clamps distance to >= 1.0 m to avoid log10(0) and near-field singularities
        """
        # Use only position columns (first 2) for path loss calculation
        pos = x[:, :2] if x.dim() > 1 and x.shape[1] > 2 else x

        # Euclidean distance (meters) with stability guards
        d = torch.norm(pos - self.theta, p=2, dim=1) + 1e-6
        d = torch.clamp(d, min=1.0)

        L = self.gamma * 10.0 * torch.log10(d)
        fobs = self.P0 - L.unsqueeze(1)
        return fobs
    def get_theta(self):
        """
        Retrieves the learned theta (position parameter).

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.theta
    
    def get_P0(self):
        """
        Retrieves the learned theta P0 (transmit power).

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.P0
    
    def get_gamma(self):
        """
        Retrieves the learned gamma (path loss exponent).

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.gamma

class Net_augmented(nn.Module):
    """
    A neural network model augmented with a pathloss model for predicting signal strength based on position.

    This model combines:
    - A pathloss model based on physics (parameters: theta, gamma, P0).
    - A neural network for learning residual components.
    
    The output is a weighted combination: w_PL * f_PL(x) + w_NN * f_NN(x)
    where weights are computed via softmax for stability.

    UPDATED: Removed unused BatchNorm, added get_physics_weight() for monitoring.

    Attributes:
    - theta (nn.Parameter): A trainable parameter representing the jammer's estimated position.
    - gamma (nn.Parameter): A trainable parameter representing the path loss exponent.
    - P0 (nn.Parameter): A trainable parameter representing the reference power level.
    - w (nn.Parameter): Fusion weight logits [w_PL_logit, w_NN_logit]
    - input_dim (int): Dimension of the input features.
    - output_dim (int): Dimension of the final output layer.
    """

    def __init__(self, input_dim, layer_wid, nonlinearity, gamma=2, theta0=None, P0_init=None):
        """
        Initializes the Net_augmented model with both a neural network and a pathloss model.

        Parameters:
        - input_dim (int): The dimension of the input features.
        - layer_wid (list): List specifying the number of neurons in each hidden layer.
        - nonlinearity (str): Activation function used in the neural network ('relu', 'sigmoid', 'softplus', 'tanh', 'leaky_relu').
        - gamma (float): Initial value for the path loss exponent (default: 2).
        - theta0 (list or None): Initial position estimate for the jammer (default: None).
        - P0_init (float or None): Initial value for P0 (reference power at 1m).
            If None, defaults to 10 (Jaramillo's original).
            Recommended: estimate from data using estimate_P0_from_data().
        """
        super().__init__()
        self.theta = nn.Parameter(torch.zeros((2))) if theta0 is None else nn.Parameter(torch.tensor(theta0))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        
        # P0 initialization: use provided value or default to 10 (Jaramillo's original)
        if P0_init is None:
            P0_init = 10.0
        self.P0 = nn.Parameter(torch.tensor(float(P0_init), dtype=torch.float32))
        
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        
        # Normalization layers (LayerNorm is stable for small batches / FL)
        self.normalization = nn.LayerNorm(input_dim)
        # NOTE: BatchNorm was previously defined but NEVER USED in forward()
        # It has been removed to clean up the code (reviewer concern E.1)
        
        self.dropout = nn.Dropout(p=0.2)
        self.fc_layers = nn.ModuleList()

        # Create input layer
        self.fc_layers.append(nn.Linear(in_features=input_dim, out_features=layer_wid[0]))

        # Create hidden layers
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], out_features=layer_wid[i + 1]))

        # Apply Xavier Initialization
        self.initialize_weights()
        

        # LayerNorm per hidden-layer input (stable for batch size 1 / FL)
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(layer.in_features) for layer in self.fc_layers[:-1]])
        # Set the activation function
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: torch.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        
        # Fusion weight logits: [w_PL_logit, w_NN_logit]
        # Softmax is applied in forward() to get actual weights
        # Initialize with physics bias (w_PL > w_NN initially)
        self.w = nn.Parameter(torch.tensor([0.8, 0.2], requires_grad=True))

    
            
    def initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):  # Apply only to Linear layers
                torch.nn.init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
                if layer.bias is not None:  # Initialize bias as zero
                    torch.nn.init.zeros_(layer.bias)
            
    def forward(self, x):
        """
        Forward pass for the augmented model.

        Parameters:
        - x (Tensor): Input tensor.

        Outputs:
        - (Tensor): Combined output based on the model mode.
        """
        w_PL, w_NN = torch.softmax(self.w, dim=0)
        
        return w_PL*self.forward_PL(x) + w_NN*self.forward_NN(x)
    
    def forward_NN(self, x):
        """
        Forward pass for the neural network branch.

        Parameters:
        - x (Tensor): Input tensor with dimensions N x input_dim (N = batch size).

        Outputs:
        - (Tensor): Output tensor after passing through the network.
        """
        # Input normalization (LayerNorm is stable for small batches / FL)
        x = self.normalization(x)

        # Hidden layers
        for fc_layer, norm in zip(self.fc_layers[:-1], self.hidden_norms):
            x = norm(x)
            x = self.nonlinearity(fc_layer(x))
            x = self.dropout(x)

        # Output layer
        return self.fc_layers[-1](x)


    def forward_PL(self, x):
        """
        Forward pass for the path-loss (physics) branch.

        Uses only position columns (x_enu, y_enu) in the first 2 dims.

        Model:
            RSSI = P0 - 10*gamma*log10(d)

        Numerical stability:
            - Adds a small epsilon to distance
            - Clamps distance to >= 1.0 m to avoid log10(0) and near-field singularities
        """
        # Use only position columns (first 2) for path loss calculation
        pos = x[:, :2] if x.dim() > 1 and x.shape[1] > 2 else x

        # Euclidean distance (meters) with stability guards
        d = torch.norm(pos - self.theta, p=2, dim=1) + 1e-6
        d = torch.clamp(d, min=1.0)

        L = self.gamma * 10.0 * torch.log10(d)
        fobs = self.P0 - L.unsqueeze(1)
        return fobs
    
    def get_layers(self):
        """
        Retrieves the input and output dimensions of all layers.

        Outputs:
        - (list): A list containing the input and output dimensions of all layers.
        """
        L = len(self.fc_layers)
        layers = (L + 1) * [0]
        layers[0] = self.fc_layers[0].in_features
        for i in range(L):
            layers[i + 1] = self.fc_layers[i].out_features
        return layers

    def get_param(self):
        """
        Returns a flattened tensor of all parameters in the network.

        Outputs:
        - (Tensor): A concatenated tensor of all network parameters.
        """
        P = torch.tensor([])
        for p in self.parameters():
            a = p.clone().detach().requires_grad_(False).reshape(-1)
            P = torch.cat((P, a))
        return P

    def get_theta(self):
        """
        Retrieves the learned theta (position parameter).

        Outputs:
        - (Tensor): The learned theta.
        """
        return self.theta
    
    def get_P0(self):
        """
        Retrieves the learned theta P0 (transmit power).

        Outputs:
        - (Tensor): The learned P0.
        """
        return self.P0
    
    def get_gamma(self):
        """
        Retrieves the learned gamma (path loss exponent).

        Outputs:
        - (Tensor): The learned gamma.
        """
        return self.gamma
    
    def get_physics_weight(self) -> float:
        """
        Get the current physics branch weight (after softmax).
        
        This is useful for monitoring whether the NN is dominating.
        If w_PL << w_NN, theta may not be identifiable (reviewer concern A.3).
        
        Returns:
            float: Physics weight in [0, 1]
        """
        with torch.no_grad():
            w_softmax = torch.softmax(self.w, dim=0)
            return float(w_softmax[0].item())
    
    def get_nn_weight(self) -> float:
        """
        Get the current neural network branch weight (after softmax).
        
        Returns:
            float: NN weight in [0, 1]
        """
        with torch.no_grad():
            w_softmax = torch.softmax(self.w, dim=0)
            return float(w_softmax[1].item())


# ==================== Utility Functions ====================

def estimate_P0_from_data(rssi_values, positions=None, gamma=2.0, method='median'):
    """
    Estimate initial P0 from RSSI data for better model initialization.
    
    The path loss model is: RSSI = P0 - 10*γ*log10(d)
    Rearranging: P0 = RSSI + 10*γ*log10(d)
    
    This function provides a data-driven initialization for P0, improving
    convergence speed compared to Jaramillo's fixed P0=10 initialization.
    
    Args:
        rssi_values: Array of RSSI measurements (dBm)
        positions: Array of (x, y) positions in ENU coordinates. 
                   If None, uses RSSI statistics only.
        gamma: Path loss exponent (default 2.0 for free space)
        method: Estimation method:
            - 'median': Median of RSSI + distance correction (robust, recommended)
            - 'max': Max RSSI + small offset (assumes closest point is ~3m)
            - 'regression': Linear regression on log-distance (requires positions)
    
    Returns:
        float: Estimated P0 value in dBm
    
    Example:
        >>> from model import Net_augmented, estimate_P0_from_data
        >>> 
        >>> # Load your data
        >>> rssi = df['RSSI'].values
        >>> positions = df[['x_enu', 'y_enu']].values
        >>> 
        >>> # Estimate P0 from data
        >>> P0_init = estimate_P0_from_data(rssi, positions, gamma=2.0)
        >>> print(f"Estimated P0: {P0_init:.2f} dBm")
        >>> 
        >>> # Create model with estimated P0
        >>> model = Net_augmented(
        ...     input_dim=4,
        ...     layer_wid=[64, 32, 1],
        ...     nonlinearity='relu',
        ...     gamma=2.0,
        ...     theta0=theta_init,
        ...     P0_init=P0_init
        ... )
    
    Note:
        Jaramillo's original implementation uses P0=10 (hardcoded).
        This data-driven approach typically provides faster convergence.
    """
    rssi_values = np.asarray(rssi_values)
    
    if method == 'max':
        # Simple: assume max RSSI is from closest point (~3m away)
        # P0 ≈ max(RSSI) + 10*γ*log10(3) ≈ max(RSSI) + 9.5 for γ=2
        P0_estimate = float(np.max(rssi_values) + 10 * gamma * np.log10(3))
        
    elif method == 'median':
        if positions is not None:
            # Estimate distance to centroid (rough jammer location estimate)
            positions = np.asarray(positions)
            centroid = positions.mean(axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            distances = np.maximum(distances, 1.0)  # Avoid log(0)
            
            # P0 = RSSI + 10*γ*log10(d) for each sample
            P0_estimates = rssi_values + 10 * gamma * np.log10(distances)
            P0_estimate = float(np.median(P0_estimates))
        else:
            # Without positions, use max RSSI method
            P0_estimate = float(np.max(rssi_values) + 10 * gamma * np.log10(3))
    
    elif method == 'regression':
        if positions is None:
            raise ValueError("Positions required for regression method")
        
        positions = np.asarray(positions)
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        distances = np.maximum(distances, 1.0)
        
        # Linear regression: RSSI = P0 - 10*γ*log10(d)
        # y = a + b*x where y=RSSI, x=log10(d), a=P0, b=-10*γ
        log_distances = np.log10(distances)
        
        # Simple linear regression
        x_mean = np.mean(log_distances)
        y_mean = np.mean(rssi_values)
        
        numerator = np.sum((log_distances - x_mean) * (rssi_values - y_mean))
        denominator = np.sum((log_distances - x_mean) ** 2)
        
        if denominator > 0:
            slope = numerator / denominator  # This is -10*γ
            intercept = y_mean - slope * x_mean  # This is P0
            P0_estimate = float(intercept)
        else:
            # Fallback if regression fails
            P0_estimate = float(np.max(rssi_values) + 10)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max', 'median', or 'regression'")
    
    return P0_estimate