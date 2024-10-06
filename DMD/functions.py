import torch as pt

def find_optimal_rank(s, thr):
    """
    Function that computes the optimal rank by evaluating singular values, based on chosen threshold.

    Parameters:
        s (torch.Tensor): Tensor containing singular values.
        thr (float): Chosen threshold to truncate singular values.

    Returns:
        optimal_rank (int): Computed optimal rank based on singular values and threshold.

    Raises:
        ValueError: If chosen threshold is negative.
        ValueError: If chosen threshold is greater than 100.
        
    """
    if thr <= 0:
        raise ValueError("Threshold must be positive")

    elif thr > 100:
        raise ValueError("Threshold must be less or equal than 100")

    s_relative = (s / s.sum() * 100)
    s_cumsum = pt.cumsum(s_relative, dim=0)

    optimal_rank = pt.where(s_cumsum >= thr)[0][0].item()
    return optimal_rank
