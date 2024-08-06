import torch


class LinearModel(torch.nn.Module):
    def __init__(self, num_joints, num_hidden_layers, hidden_size) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(num_joints, hidden_size),
            torch.nn.Sigmoid(),
            *(
                num_hidden_layers
                * (torch.nn.Linear(hidden_size, hidden_size), torch.nn.Sigmoid())
            ),
            torch.nn.Linear(hidden_size, 5)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sequential(x)
        x[:, -1] = self.sigmoid(x[:, -1])
        return x
