import torch as pt
from flowtorch.data import mask_box
from DMD.data_loader import load_data
from DMD.config import FIELD_NAME, MASK_LOWER_BOUND, MASK_UPPER_BOUND

def process_data():
    """
    Function that takes loaded data and process them.

    Returns:
        mask (torch.BoolTensor): Matrix of 0s and 1s to restrict data.
        t_steps (list): List of time steps used.
        dt (float): Time interval between adjacent time steps.
        data_matrix (torch.FloatTensor): Matrix of vorticity values.
        
    """
    times, pts, loader = load_data()
    mask = mask_box(pts, lower=MASK_LOWER_BOUND, upper=MASK_UPPER_BOUND)

    # In case the large dataset is used, only times greater than 4s are selected
    # The reason is that vortex shedding is complete after 4 seconds
    min_time_threshold = max(4.0, float(times[0]))
    t_steps = [t for t in times if float(t) >= min_time_threshold]
    
    dt = round(float(t_steps[1]) - float(t_steps[0]), 3)
    data_matrix = pt.zeros(pt.count_nonzero(mask), len(t_steps))

    for idx, t in enumerate(t_steps):
        snapshots = loader.load_snapshot(FIELD_NAME, t)
      
        # Vorticity is defined as the curl of velocity, it has non-zero values only along z-axis
        data_matrix[:, idx] = pt.masked_select(snapshots[:, 2], mask)

    if data_matrix.dtype not in (pt.complex64, pt.complex128):
        data_matrix = data_matrix.type(pt.cfloat)
            
    return mask, t_steps, dt, data_matrix
