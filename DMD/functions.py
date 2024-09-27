def find_optimal_rank(s, thr):
        tot = 0
        s_relative = [s_i / s.sum() * 100 for s_i in s]   
    
        for i in range(len(s_relative)):
            tot += s_relative[i]
    
            if tot >= thr:
                optimal_rank = i
                break
    
        return optimal_rank
