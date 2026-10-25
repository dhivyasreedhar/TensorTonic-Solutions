import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    d_k=query.shape[-1]
    scores=torch.matmul(query,key.transpose(-2,-1))/(d_k**0.5)

    if mask is not None:
        scores=scores.masked_fill(mask,float("-inf"))

    weights=torch.softmax(scores,dim=-1)

    return torch.matmul(weights,value)
    
