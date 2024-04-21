import torch


class AutoRec(torch.nn.Module):
    def __init__(self, len_vec: int, num_hidden: int):
        super().__init__()
        # encoder layer g(VR + b1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(len_vec, num_hidden, bias=True),
            torch.nn.Sigmoid(),
        )

        # decoder layer f(W * g + b2)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, len_vec, bias=True),
        )

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_vec)
        pred = self.decoder(hidden)
        if self.training:
            return pred.float() * torch.sign(input_vec)  # mask unobserved inputs (zeros; no negative ratings)
        return pred
