import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, fc_units, output_size, target_length, device):
        super(MLP, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_units)
        self.fc3 = nn.Linear(fc_units, output_size)
        self.target_length = target_length

    def forward(self, input):
        input = input[:, -1, :].unsqueeze(1)
        outputs = torch.zeros([input.shape[0], self.target_length, input.shape[2]]).to(self.device)
        for di in range(self.target_length):
            output = self.fc1(input)
            output = F.relu(output)
            output = self.fc2(output)
            output = F.relu(output)
            output = self.fc3(output)
            input = output
            outputs[:, di:di+1, :] = output

        return outputs
    

