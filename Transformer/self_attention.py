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
        softmax = nn.Softmax(dim=2)
        attention_weight = softmax(
            torch.matmul(query, torch.transpose(key, -1, -2))
        )
        weighted_sum = torch.matmul(attention_weight, value)
        return weighted_sum, attention_weight


input_dim = 8
batch_size = 16
seq_legenth = 2


x_input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]], device=device)
x_input = torch.unsqueeze(x_input, dim=0)
self_attention = SelfAttention(input_dim)
print(self_attention(x_input)[1].cpu().detach().numpy().reshape((2, 2)))
