import torch
import torch.nn as nn


class QModel(nn.Module):
    def __init__(self, embedding_size: int = 20):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, user: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        """Evaluate the user-doc Q model

        Args:
            user (torch.Tensor): User embedding of shape (batch_size,
                embedding_size).
            doc (torch.Tensor): Doc embeddings of shape (batch_size,
                num_docs, embedding_size).

        Returns:
            score (torch.Tensor): q_values of shape (batch_size, num_docs+1).
        """
        batch_size, num_docs, embedding_size = doc.shape
        doc_flat = doc.view((batch_size * num_docs, embedding_size))
        user_repeated = user.repeat(num_docs, 1)
        x = torch.cat([user_repeated, doc_flat], dim=1)
        x = self.layers(x)
        return torch.cat(
            [x.view((batch_size, num_docs)), torch.zeros((batch_size, 1))], dim=1
        )


if __name__ == "__main__":
    m = QModel()
    user = torch.randn((5, 20))
    doc = torch.randn((5, 10, 20))
    q_values = m(user, doc)
