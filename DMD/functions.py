def find_optimal_rank(s, thr):
    if thr <= 0:
        raise ValueError("Threshold must be positive")

    elif thr > 100:
        raise ValueError("Threshold must be less or equal than 100")

    s_relative = (s / s.sum() * 100)
    s_cumsum = pt.cumsum(s_relative, dim=0)

    optimal_rank = pt.where(s_cumsum >= thr)[0][0].item()
    return optimal_rank
