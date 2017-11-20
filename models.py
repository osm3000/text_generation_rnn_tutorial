import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
class SimpleGRU(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(SimpleGRU, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.gru_layer = nn.GRUCell(input_size=in_size, hidden_size=hidden_size)
        self.dense = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, in_seq):
        """
        The in_seq should have the shape (batch size X sequence length X sequence dimension)
        """
        timesteps = in_seq.size(1)
        h_t = Variable(torch.zeros(in_seq.size(0), self.hidden_size), requires_grad=False)
        if torch.cuda.is_available():
            h_t = h_t.cuda()

        # Apply the GRU first, step by step
        for i in range(timesteps):
            h_t = self.gru_layer(in_seq[:, i, :], h_t)

        # Then, apply the dense layer
        dense_output = self.dense(h_t)

        # Finally, apply the softmax function
        final_output = F.log_softmax(dense_output)
        # final_output = F.softmax(dense_output)

        return final_output
