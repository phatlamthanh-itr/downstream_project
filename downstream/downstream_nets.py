import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# ==================Transformer Decoder=====================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerDecoder(nn.Module):
    def __init__(self, input_seq_length=2500, reduced_seq_length=80, feature_dim=320, 
                 num_layers=2, num_heads=4, ff_dim=512, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.input_seq_length = input_seq_length
        self.output_seg_length = reduced_seq_length
        
        # Downsampling from 2500 -> 80
        self.downsample = nn.Conv1d(feature_dim, feature_dim, kernel_size= self.input_seq_length // self.output_seg_length, 
                                    stride=self.input_seq_length // self.output_seg_length)  
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(feature_dim, max_len=reduced_seq_length)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, 
                                                   dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Projection
        self.output_layer = nn.Linear(feature_dim, 1)  
    

    def setup_dataloader(self, data: np.array, label: np.array, batch_size: int, train: bool) -> DataLoader:

        data_tensor = torch.from_numpy(data).float()
        mean = data_tensor.mean(dim=(0, 1), keepdim=True)  
        std = data_tensor.std(dim=(0, 1), keepdim=True) + 1e-6  

        data_tensor = (data_tensor - mean) / std
        label_tensor = torch.from_numpy(label).float()
        
        dataset = TensorDataset(data_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader
    


    def forward(self, x, memory):
        x = self.downsample(x.transpose(1, 2)).transpose(1, 2)   # (batch_size, 2500, 320) -> (batch_size, 80, 320)

        # Positional Encoding
        x = self.positional_encoding(x)  # (batch_size, 2500, 320) -> (batch_size, 80, 320)

        # Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)  # (batch_size, 80, feature_dim)

        # FCN
        output = self.output_layer(output).squeeze(-1)  # (batch_size, 80)

        return output

# ===================== LSTM ===========================
class LSTMDecoder(nn.Module):  
    """
    LSTM-base decoder (work with 80-length input sequence)
    """
    def __init__(self, input_seq_length=2500, feature_dim=320, hidden_dim=32, num_layers=2, output_length=80, bidirectional=True):
            super(LSTMDecoder, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # Downsampling (Conv1D to reduce from 2500 to 80)
            # self.downsample = nn.Conv1d(input_dim, input_dim, kernel_size=input_seq_length//output_length, stride=input_seq_length//output_length, padding=1)
            # Multi-layer Downsampling using Conv1d
            self.downsample = nn.Sequential(
                nn.Conv1d(feature_dim, 256, kernel_size=9, stride=4, padding=3),  # (bs, 256, 625)
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),

                nn.Conv1d(256, 128, kernel_size=7, stride=4, padding=2),  # (bs, 128, 156)
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),

                nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=4),  # (bs, 64, 80)
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),

            )
            # First LSTM layer
            self.lstm1 = nn.LSTM(hidden_dim * self.num_directions, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
            # # Second LSTM layer 
            self.lstm2 = nn.LSTM(hidden_dim * self.num_directions, 1, num_layers=1, batch_first=True, bidirectional=True)

            self.last_conv1d = nn.Conv1d(hidden_dim * 2 if bidirectional else hidden_dim, 1, kernel_size=1, stride=1, padding=0)
            
            # FCN
            self.fc = nn.Linear(output_length * self.num_directions, output_length)
            self.apply(self.init_weights)
    
    def init_weights(self, module):
        """Initialize weights for Conv1D, LSTM, and Linear layers."""
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.orthogonal_(param) 
                elif 'bias' in name:
                    nn.init.constant_(param, 0)  
        
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)   
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def setup_dataloader(self, data: np.array, label: np.array, batch_size: int, train: bool) -> DataLoader:

        data_tensor = torch.from_numpy(data).float()
        # mean = data_tensor.mean(dim=(0), keepdim=True)
        # std = data_tensor.std(dim=(0), keepdim=True) + 1e-6

        # data_tensor = (data_tensor - mean) / std
        label_tensor = torch.from_numpy(label).float()
        # print("Normalize dataset done")
        
        dataset = TensorDataset(data_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader


    def forward(self, x, mask):
        batch_size = x.size(0)

        # Downsampling (2500 → 80)
        x = x.permute(0, 2, 1)  # Shape: (batch, 320, 2500)
        x = self.downsample(x)  # Shape: (batch, 320, 80)
        x = x.permute(0, 2, 1)  # Shape: (batch, 80, 320)
        

        x, _ = self.lstm1(x)                   # Shape: (batch, 80, hidden_dim)
        x, _ = self.lstm2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class LSTMDecoder2500(nn.Module):  
    """
    LSTM-based decoder (works with 2500-length input sequence)
    
    """
    def __init__(self, input_seq_length=2500, feature_dim=320, hidden_dim=32, num_layers=1, output_length=80, bidirectional=False):
            super(LSTMDecoder2500, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # Multi-layer Downsampling using Conv1d
            self.downsample = nn.Sequential(
                nn.Conv1d(feature_dim, 256, kernel_size=9, stride=4, padding=3),  # (bs, 256, 625)
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),

                nn.Conv1d(256, 128, kernel_size=7, stride=4, padding=2),  # (bs, 128, 156)
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),

                nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=4),  # (bs, 64, 80)
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),

            )
            # First LSTM layer
            self.lstm1 = nn.LSTM(hidden_dim * self.num_directions,  hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
            # # Second LSTM layer 
            self.lstm2 = nn.LSTM(hidden_dim * self.num_directions, 1, num_layers=1, batch_first=True, bidirectional=bidirectional)
            
            # FCN
            self.fc = nn.Linear(output_length * self.num_directions, input_seq_length)
            self.apply(self.init_weights)
    
    def init_weights(self, module):
        """Initialize weights for Conv1D, LSTM, and Linear layers."""
        if isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.orthogonal_(param) 
                elif 'bias' in name:
                    nn.init.constant_(param, 0)  
        
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)   
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def setup_dataloader(self, data: np.array, label: np.array, batch_size: int, train: bool) -> DataLoader:

        data_tensor = torch.from_numpy(data).float()
        # mean = data_tensor.mean(dim=(0), keepdim=True)
        # std = data_tensor.std(dim=(0), keepdim=True) + 1e-6

        # data_tensor = (data_tensor - mean) / std
        label_tensor = torch.from_numpy(label).float()
        # print("Normalize dataset done")
        
        dataset = TensorDataset(data_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader


    def forward(self, x, mask):
        batch_size = x.size(0)

        # Downsampling (2500 → 80)
        x = x.permute(0, 2, 1)  # Shape: (batch, 320, 2500)
        x = self.downsample(x)  # Shape: (batch, 320, hidden_dim)
        x = x.permute(0, 2, 1)  # Shape: (batch, 80, hidden_dim)
        
        x, _ = self.lstm1(x)                   # Shape: (batch, 80, hidden_dim)
        x, _ = self.lstm2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

 
    # Example usage:
    batch_size = 16
    seq_length = 2500
    input_dim = 320
    x = torch.randn(batch_size, seq_length, input_dim)

    # model = LSTMDecoder(input_seq_length=seq_length, input_dim=input_dim, hidden_dim=32, num_layers=2, output_length=80)
    model = LSTMDecoder2500(input_seq_length=seq_length, feature_dim=input_dim, hidden_dim=64, num_layers=2, output_length=80, bidirectional=False)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    output = model(x, x)
    print(output.shape)  # Expected: (batch, 80, hidden_dim)