"""LSTM for long-term sequence prediction with attention mechanism."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for LSTM model."""
    input_size: int = 10
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    num_attention_heads: int = 4
    sequence_length: int = 12  # months
    prediction_horizon: int = 6  # months
    quantile_levels: List[float] = None
    
    def __post_init__(self):
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.5, 0.9]

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer for sequence modeling."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + context)
        
        return output, attention_weights

class MigrationLSTM(nn.Module):
    """LSTM with attention for migration sequence prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            config.input_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.dropout
        )
        
        # Quantile prediction heads
        self.quantile_heads = nn.ModuleDict({
            f'q{int(q*100)}': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, config.prediction_horizon)
            ) for q in config.quantile_levels
        })
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            lengths: Sequence lengths for padding [batch_size]
            
        Returns:
            Dictionary with quantile predictions and attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack sequences if lengths provided
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed_x)
            unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
            unpacked_out = lstm_out
        
        # Apply attention
        attended_out, attention_weights = self.attention(unpacked_out)
        
        # Global average pooling (or use last hidden state)
        if lengths is not None:
            # Use last valid output for each sequence
            pooled = torch.stack([
                attended_out[i, lengths[i]-1, :] for i in range(batch_size)
            ])
        else:
            pooled = torch.mean(attended_out, dim=1)
        
        # Generate quantile predictions
        predictions = {}
        for quantile_name, head in self.quantile_heads.items():
            predictions[quantile_name] = head(pooled)
        
        # Generate uncertainty estimates
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled))
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'hidden_states': hidden
        }

class SequenceDataProcessor:
    """Data processor for sequence-based training."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scalers = {}
        self.feature_columns = []
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         target_col: str = 'flow',
                         feature_cols: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training.
        
        Args:
            df: DataFrame with time series data
            target_col: Name of target column
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (sequences, targets, lengths)
        """
        if feature_cols is None:
            feature_cols = [
                'pop_o', 'pop_d', 'chirps_spi3_o', 'era5_tmax_anom_o', 'access_score_o',
                'chirps_spi3_d', 'era5_tmax_anom_d', 'access_score_d', 'acled_intensity_o', 'acled_intensity_d'
            ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        # Normalize features
        df_normalized = df.copy()
        for col in available_features + [target_col]:
            if col in df.columns:
                df_normalized[col] = self._normalize_feature(df[col], col)
        
        # Create sequences
        sequences = []
        targets = []
        lengths = []
        
        # Group by origin-destination pairs
        for (origin, dest), group in df_normalized.groupby(['origin_id', 'dest_id']):
            group = group.sort_values('period').reset_index(drop=True)
            
            if len(group) >= self.config.sequence_length + self.config.prediction_horizon:
                # Create sliding windows
                for i in range(len(group) - self.config.sequence_length - self.config.prediction_horizon + 1):
                    seq_features = group[available_features].iloc[i:i + self.config.sequence_length].values
                    seq_targets = group[target_col].iloc[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon].values
                    
                    sequences.append(seq_features)
                    targets.append(seq_targets)
                    lengths.append(self.config.sequence_length)
        
        if not sequences:
            raise ValueError("No valid sequences found")
        
        return (
            torch.tensor(np.array(sequences), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
            torch.tensor(np.array(lengths), dtype=torch.long)
        )
    
    def _normalize_feature(self, series: pd.Series, name: str) -> pd.Series:
        """Normalize feature using z-score."""
        if name not in self.scalers:
            mean = series.mean()
            std = series.std()
            self.scalers[name] = {'mean': mean, 'std': std}
        else:
            mean = self.scalers[name]['mean']
            std = self.scalers[name]['std']
        
        return (series - mean) / (std + 1e-8)

def train_lstm_model(model: MigrationLSTM,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    epochs: int = 50,
                    learning_rate: float = 1e-3,
                    device: str = 'cpu') -> Dict:
    """Train the LSTM model with quantile loss."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_sequences, batch_targets, batch_lengths in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_sequences, batch_lengths)
            predictions = outputs['predictions']
            
            # Calculate quantile loss
            loss = 0.0
            for i, (quantile_name, pred) in enumerate(predictions.items()):
                q = model.config.quantile_levels[i]
                loss += quantile_loss(pred, batch_targets, q)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_sequences, batch_targets, batch_lengths in val_loader:
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                batch_lengths = batch_lengths.to(device)
                
                outputs = model(batch_sequences, batch_lengths)
                predictions = outputs['predictions']
                
                loss = 0.0
                for i, (quantile_name, pred) in enumerate(predictions.items()):
                    q = model.config.quantile_levels[i]
                    loss += quantile_loss(pred, batch_targets, q)
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }

def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantile: float) -> torch.Tensor:
    """Calculate quantile loss (pinball loss)."""
    errors = targets - predictions
    loss = torch.maximum(
        (quantile - 1) * errors,
        quantile * errors
    )
    return torch.mean(loss)

def create_synthetic_sequence_data(n_samples: int = 1000, 
                                 sequence_length: int = 12,
                                 prediction_horizon: int = 6) -> pd.DataFrame:
    """Create synthetic sequence data for testing."""
    np.random.seed(42)
    
    data = []
    for sample in range(n_samples):
        origin = f"admin_{sample % 5}"
        dest = f"admin_{(sample + 1) % 5}"
        
        # Generate time series
        periods = pd.date_range('2020-01', periods=sequence_length + prediction_horizon, freq='M')
        
        for i, period in enumerate(periods):
            # Generate features with temporal dependencies
            pop_o = 100000 + np.random.normal(0, 10000)
            pop_d = 150000 + np.random.normal(0, 15000)
            
            # Add seasonal patterns
            seasonal = np.sin(2 * np.pi * i / 12)
            chirps_spi3_o = seasonal + np.random.normal(0, 0.5)
            era5_tmax_anom_o = seasonal * 0.5 + np.random.normal(0, 0.3)
            
            access_score_o = 0.5 + np.random.normal(0, 0.2)
            chirps_spi3_d = seasonal * 0.8 + np.random.normal(0, 0.4)
            era5_tmax_anom_d = seasonal * 0.3 + np.random.normal(0, 0.2)
            access_score_d = 0.6 + np.random.normal(0, 0.15)
            
            # Conflict with some persistence
            if i == 0:
                acled_intensity_o = np.random.exponential(0.5)
                acled_intensity_d = np.random.exponential(0.3)
            else:
                acled_intensity_o = 0.7 * acled_intensity_o + 0.3 * np.random.exponential(0.5)
                acled_intensity_d = 0.7 * acled_intensity_d + 0.3 * np.random.exponential(0.3)
            
            # Generate flow with complex dependencies
            base_flow = (np.log(pop_o) * 0.3 + 
                        chirps_spi3_o * 0.2 + 
                        acled_intensity_o * 0.4 - 
                        access_score_o * 0.1 +
                        access_score_d * 0.2)
            
            flow = max(0, base_flow + np.random.normal(0, 0.1))
            
            data.append({
                'origin_id': origin,
                'dest_id': dest,
                'period': period.strftime('%Y-%m'),
                'pop_o': pop_o,
                'pop_d': pop_d,
                'chirps_spi3_o': chirps_spi3_o,
                'era5_tmax_anom_o': era5_tmax_anom_o,
                'access_score_o': access_score_o,
                'chirps_spi3_d': chirps_spi3_d,
                'era5_tmax_anom_d': era5_tmax_anom_d,
                'access_score_d': access_score_d,
                'acled_intensity_o': acled_intensity_o,
                'acled_intensity_d': acled_intensity_d,
                'flow': flow
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test LSTM model
    print("Testing LSTM sequence model...")
    
    # Create synthetic data
    df = create_synthetic_sequence_data(100, 12, 6)
    print(f"Created synthetic dataset with {len(df)} samples")
    
    # Create model config
    config = ModelConfig(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        sequence_length=12,
        prediction_horizon=6
    )
    
    # Create model
    model = MigrationLSTM(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    processor = SequenceDataProcessor(config)
    sequences, targets, lengths = processor.prepare_sequences(df)
    
    print(f"Prepared {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Test model forward pass
    with torch.no_grad():
        outputs = model(sequences[:5], lengths[:5])
        print(f"Model outputs: {list(outputs.keys())}")
        print(f"Quantile predictions shape: {outputs['predictions']['q10'].shape}")
        print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    
    print("LSTM model test completed successfully!")
