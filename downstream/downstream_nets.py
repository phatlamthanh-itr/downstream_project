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


# ============================================ LSTM 2500 DECODER =============================================================

class ChannelSELayer1d(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer1d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, sequence_length, H)
        :return: output tensor
        """
        batch_size, num_channels, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))

        # a, b = squeeze_tensor.size()
        # output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1))
        return input_tensor * fc_out_2.view(batch_size, num_channels, 1)


class SpatialSELayer1d(nn.Module):

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer1d, self).__init__()
        self.conv = nn.Conv1d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, seq_len, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        return input_tensor * squeeze_tensor.view(batch_size, 1, a)


class ChannelSpatialSELayer1d(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param seq_length: No of input channels
        """
        super(ChannelSpatialSELayer1d, self).__init__()
        self.cSE = ChannelSELayer1d(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer1d(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, seg_len, W)
        :return: output_tensor
        """
        return self.sSE(self.cSE(input_tensor))
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
                nn.Conv1d(feature_dim, feature_dim, kernel_size=9, stride=1, padding=3),  # (bs, feature dim, 625)
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(),

                nn.Conv1d(feature_dim, 256, kernel_size=9, stride=4, padding=3),  # (bs, 256, 625)
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                # nn.GELU(),


                nn.Conv1d(256, 128, kernel_size=7, stride=4, padding=2),  # (bs, 128, 156)
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                # nn.GELU(),

                nn.Conv1d(128, self.hidden_dim * self.num_directions, kernel_size=5, stride=2, padding=4),  # (bs, 64, 80)
                nn.BatchNorm1d(self.hidden_dim * self.num_directions),
                nn.LeakyReLU(),
                # nn.GELU(),
            )

            # SE blocks
            self.se_blocks = ChannelSpatialSELayer1d(num_channels=output_length)


            # # First LSTM layer
            self.lstm1 = nn.LSTM(self.hidden_dim * self.num_directions,  self.hidden_dim, num_layers = self.num_layers, batch_first=True, bidirectional=bidirectional)
            # # # Second LSTM layer 
            self.lstm2 = nn.LSTM(self.hidden_dim * self.num_directions, 1, num_layers=1, batch_first=True, bidirectional=bidirectional)            
            
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
        x = self.downsample(x)  # Shape: (batch, 320, hidden_dim)   (Conv1d blocks)
        x = x.permute(0, 2, 1)  # Shape: (batch, 80, hidden_dim)
        
        # SE-block
        x = self.se_blocks(x)

        x, _ = self.lstm1(x)                   # Shape: (batch, 80, hidden_dim)
        x, _ = self.lstm2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x




# ============================================ RESNET1D + LSTM DECODER ===================================================================


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d with padding same
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d with SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.GELU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.relu1 = nn.ReLU()
        self.relu2 = nn.GELU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1DEncoder(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer
        n_block: number of blocks
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1DEncoder, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)

        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # Residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)
        # out = out.mean(-1)
        return out
    


class ResNetLSTM(nn.Module):
    def __init__(self, 
                 input_seq_length=2500, 
                 feature_dim=320, 
                 hidden_dim=32,
                 num_layers=1, 
                 output_length=80, 
                 bidirectional=False,
                 kernel_size=3, 
                 stride=1, 
                 groups=1, 
                 n_block=4, 
                 downsample_gap=2, 
                 increasefilter_gap=4, 
                 use_bn=True, 
                 use_do=True, 
                 verbose=False):
        super(ResNetLSTM, self).__init__()
        self.hidden_dim = hidden_dim     # It will be the base_filters args for ResNet1d Encoder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.feature_dim = feature_dim
        self.input_seq_length = input_seq_length
        self.output_length = output_length
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # ResNet
        self.resnet_encoder = ResNet1DEncoder(in_channels=self.feature_dim, 
                                              base_filters=self.hidden_dim,
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              groups=self.groups,
                                              n_block=self.n_block,
                                              downsample_gap=self.downsample_gap,
                                              increasefilter_gap=self.increasefilter_gap,
                                              use_bn=self.use_bn,
                                              verbose=self.verbose)
        
        # LSTM decoder
        # First LSTM layer
        self.lstm1 = nn.LSTM(self.hidden_dim * self.num_directions,  self.hidden_dim , self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        # # Second LSTM layer 
        self.lstm2 = nn.LSTM(self.hidden_dim * self.num_directions, 1, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
        
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

        # ResNet encoder
        x = x.permute(0, 2, 1)  # Shape: (batch, 320, 2500)
        x = self.resnet_encoder(x)  # Shape: (batch, hidden_dim, 2500)
        x = x.permute(0, 2, 1)  # Shape: (batch, 2500, hidden_dim)


        # LSTM decoder
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.view(batch_size, -1)
        # x = self.fc(x)
        return x



if __name__ == "__main__":

 
    # Example usage:
    batch_size = 16
    seq_length = 2500
    feature_dim = 320
    x = torch.randn(batch_size, seq_length, feature_dim)


    # model = LSTMDecoder(input_seq_length=seq_length, input_dim=input_dim, hidden_dim=32, num_layers=2, output_length=80)
    model = LSTMDecoder2500(input_seq_length=seq_length, feature_dim=feature_dim, hidden_dim=32, num_layers=1, output_length=80, bidirectional=True)

    # model = ResNet1DEncoder(in_channels=feature_dim, base_filters=64, kernel_size=3, stride=1, groups=1, n_block=4, verbose=True)
    # model = ResNetLSTM(input_seq_length=2500, feature_dim=320, hidden_dim=64, num_layers=1, output_length=80)
    # model = ChannelSpatialSELayer1d(num_channels = seq_length)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    output = model(x,x)
    print(output.shape)