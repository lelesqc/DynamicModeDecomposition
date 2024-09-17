import torch as pt
from data_loader import load_data, apply_mask
from config import MIN_TIME, FIELD_NAME

def process_data():
    loader, times, fields, pts = load_data()

    # Vortex shedding is complete after 4s, keep only time steps between 4s and 10s
    t_steps = [t for t in times if float(t) >= MIN_TIME]
    dt = round(float(t_steps[1]) - float(t_steps[0]), 3)

    mask = apply_mask(pts)
    data_matrix = pt.zeros((pt.count_nonzero(mask), len(t_steps)), dtype=pt.float32)

    for idx, t in enumerate(t_steps):
        snapshots = loader.load_snapshot(FIELD_NAME, t)
      
        # Vorticity is defined as the curl of velocity, it has non-zero values only along z-axis
        data_matrix[:, idx] = pt.masked_select(snapshots[:, 2], mask)
            
    return t_steps, dt, mask, data_matrix, pts
