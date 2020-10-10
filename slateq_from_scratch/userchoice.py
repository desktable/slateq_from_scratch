import torch
import torch.nn as nn


class UserChoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, user: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        """Evaluate the user choice model

        Args:
            user (torch.Tensor): User embedding of shape (batch_size,
                embedding_size).
            doc (torch.Tensor): Doc embeddings of shape (batch_size,
                num_docs, embedding_size).

        Returns:
            score (torch.Tensor): logits of shape (batch_size,
                num_docs + 1), where the last dimension represents no_click.
        """
        batch_size = user.shape[0]
        s = torch.einsum("be,bde->bd", user, doc)
        s = s * self.a
        s = torch.cat([s, self.b.expand((batch_size, 1))], dim=1)
        return s


if __name__ == "__main__":
    m = UserChoiceModel()
    user = torch.randn((5, 20))
    doc = torch.randn((5, 10, 20))
    scores = m(user, doc)
    prob = nn.Softmax(dim=1)(scores)
