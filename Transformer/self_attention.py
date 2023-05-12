import torch
import torch.nn as nn
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.W_query = nn.Linear(input_dim, input_dim, bias=False, device=device)
        self.W_key = nn.Linear(input_dim, input_dim, bias=False, device=device)
        self.W_value = nn.Linear(input_dim, input_dim, bias=False, device=device)

    def forward(self, input_x):
        query = self.W_query(input_x)
        key = self.W_key(input_x)
        value = self.W_value(input_x)
        softmax = nn.Softmax(dim=1)
        attention_weight = softmax(
            torch.matmul(query, torch.transpose(key, -1, -2))
        )
        weighted_sum = torch.matmul(attention_weight, value)
        return weighted_sum, attention_weight


input_dim = 128
batch_size = 16
seq_legenth = 10

x_input = torch.randn(batch_size, seq_legenth, input_dim, device=device)
self_attention = SelfAttention(input_dim)
print(self_attention(x_input)[1][0].data)
