import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.frequency_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x):
        out = [x]
        for freq in self.frequency_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

class NeRF(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4, num_layers=8, num_frequencies=10):
        super(NeRF, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim, num_frequencies)
        input_size = input_dim * (2 * num_frequencies + 1)  # Encoded input size
        
        layers = [nn.Linear(input_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))  # Density + RGB output
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x_encoded = self.positional_encoding(x)
        return self.model(x_encoded)
