import torch


@torch.no_grad()
def token_unigram_entropy_from_token_ids(tokens_1d: torch.Tensor) -> float:
    """Unigram entropy (nats) of one sample's token ids."""
    tokens_1d = tokens_1d.view(-1).to(torch.long)
    if tokens_1d.numel() == 0:
        return 0.0
    _, counts = torch.unique(tokens_1d, return_counts=True, sorted=False)
    p = counts.float() / counts.sum()
    return float(torch.special.entr(p).sum().item())
