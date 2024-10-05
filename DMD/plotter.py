import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import warnings
from data_processor import process_data

class Plotter:     
    def __init__(self, pts, mask):          
        """
        Class for visualizing data and results.

        Parameters:
            pts (torch.FloatTensor): Vertices of the grid.
            mask (torch.BoolTensor): Matrix of 0s and 1s to restrict data.
            data_matrix (torch.FloatTensor): Matrix of data, could be vorticity, velocity etc.
            time_steps (list): List of time steps of the system state.

        Methods:
            scatter_plot: Produces a scatter plot of the grid's vertices.
            plot_data(ax, data, title): Creates a filled contour plot with additional contour lines and a circle patch on the given axis.
            plot_DMD_modes(phi, mode_indices): Plots the found DMD modes.
            time_dynamics(optimal_rank, dynamics, time_steps): Plots the time evolution of each mode.
            data_reconstruction(data_matrix, reconstruction, t_idx, time_steps): Plots both original and reconstructed data for comparison.
            reconstruction_error(time_steps, mse_dmd): Plots the Mean Square Error (MSE) of reconstructed data with respect to original ones.

        """
        self.pts = pts
        self.mask = mask

    def scatter_plot(self):                   
        """
        Produces a scatter plot of the grid's vertices.

        No parameters needed.
            
        """     
        # Only 4 values are selected to enhance clarity of plots
        every = 4
        
        fig, ax = plt.subplots()
        ax.scatter(self.pts[::every, 0], self.pts[::every, 1], s = .5, alpha = self.mask[::every], c = "k")
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0.0, 2)
        
    def plot_data(self, ax, data, title):
        """
        Creates a filled contour plot with additional contour lines and a circle patch on the given axis.
        
        Parameters:
            ax (matplotlib.axes.Axes): Axis on which the plot is drawn
            data (torch.Tensor): Data array to be visualized
            title (str or Any): Title of the plot, will be converted to string if necessary
        
        Returns:
            contourf (matplotlib.contour.QuadContourSet): The filled contour plot object.     

        Raises:
            ValueError: If data shape doesn't match the number of points of the axes. 

        """
        contourf = None

        x = pt.masked_select(self.pts[:, 0], self.mask)
        y = pt.masked_select(self.pts[:, 1], self.mask)
        
        if data.size(0) != x.size(0):
            raise ValueError("Size of data must match the number of points on plot's axes.")

        ax.tricontourf(x, y, data, levels = 15, cmap = "jet")
        ax.tricontour(x, y, data, levels = 15, linewidths = 0.1, colors = 'k')
        ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color = 'k'))
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        plt.tight_layout()
    
    def plot_DMD_modes(self, phi, mode_indices):
        """
        Plots the found DMD modes.
    
        Parameters:
            phi (torch.Tensor): Tensor containing DMD modes.
            mode_indices (list): List of modes indices to be plotted.
    
        Raises:
            ValueError: If mode_indices is empty.
            IndexError: If mode_indices contains one or more invalid indices.
            
        """    
        if len(mode_indices) < 1:
            raise ValueError("At least one mode index must be provided.")

        elif any(idx > phi.size(1) for idx in mode_indices):
            raise IndexError(f"Index or indices out of bound. There are {phi.size(1)} modes that can be accessed")        
    
        thr = 1.0e-10
        phi.imag[abs(phi.imag) < thr] = 0
    
        num_rows = len(mode_indices)
    
        fig, axarr = plt.subplots(num_rows, 2, figsize=(14, 8))
        axarr = np.atleast_2d(axarr)
    
        for i, idx in enumerate(mode_indices):
            self.plot_data(axarr[i, 0], phi[:, idx].real, f"Mode {idx}, real")
            self.plot_data(axarr[i, 1], phi[:, idx].imag, f"Mode {idx}, imag")

        plt.tight_layout()

    def time_dynamics(optimal_rank, dynamics, time_steps): 
        """
        Plots the time evolution of each mode.

        Parameters:
            optimal_rank (int): Rank of the truncated matrices.
            dynamics (torch.Tensor): Tensor whose rows represent each mode evolution
            time_steps (list): List of times available.

        """      
        time_steps = [float(t) for t in time_steps]
        
        fig, axarr = plt.subplots(int(optimal_rank), 1, sharex=True)

        if optimal_rank == 1:
            axarr = [axarr]
        
        modes = list(range(0, optimal_rank))

        for i, m in enumerate(modes):
            axarr[i].plot(time_steps, dynamics[m].real, lw=1)
            axarr[i].set_ylabel(f"{m}")
            axarr[i].xaxis.set_ticks([])
            axarr[i].yaxis.set_ticks([])
            
        axarr[-1].set_xlabel(r"$t$ in $s$")
        axarr[-1].set_xlim(time_steps[0], time_steps[-1])
        axarr[0].set_title("time dynamics")

    def data_reconstruction(self, data_matrix, reconstruction, t_idx, time_steps):
        """
        Plots both original and reconstructed data for comparison.

        Parameters:
            data_matrix (torch.Tensor): Original matrix of data
            reconstruction (torch.Tensor): Reconstructed data tensor through found DMD modes
            t_idx (list): Indices of time steps to be plotted
            time_steps (list): Time steps available

        Raises:
            ValueError: If `reconstruction` and `data_matrix` have different dimensions
            ValueError: If `t_idx` is empty
    
        """ 
        if isinstance(t_idx, int):
            t_idx = [t_idx]        
        elif len(t_idx) == 0:
            raise ValueError("`t_idx` must contain at least one time-step to plot.")
        
        if reconstruction.size() != data_matrix.size():
            raise ValueError("`reconstruction` and `data_matrix` must have the same shape.")

        x = pt.masked_select(self.pts[:, 0], self.mask)
        y = pt.masked_select(self.pts[:, 1], self.mask)

        fig, axarr = plt.subplots(len(t_idx), 2, figsize = (14, 8), sharex = True, sharey = True)
        axarr = np.atleast_2d(axarr)
    
        for i, idx in enumerate(t_idx):
            self.plot_data(axarr[i, 0], data_matrix[:, idx], f"original, t = {time_steps[idx]}s")
            self.plot_data(axarr[i, 1], reconstruction[:, idx], f"DMD, t = {time_steps[idx]}s")

        plt.tight_layout()

    def reconstruction_error(time_steps, mse_dmd):    
        """
        Plots the Mean Square Error (MSE) of reconstructed data with respect to original ones.

        Parameters:
            times_steps (list): List of times available
            mse_dmd (torch.Tensor): Tensor with the computed Mean Square Error for reconstructed and original data

        Raises:
            ValueError: If `mse_dmd` has not the same length as `time_steps`
        
        """      
        time_steps = [float(t) for t in time_steps]
        
        if mse_dmd.size(0) != len(time_steps):
            raise ValueError("`mse_dmd` must be of the same size as `time_steps`.")
        
        plt.figure(figsize = (10, 6))
        plt.plot(time_steps, mse_dmd, label = "MSE")
        plt.xlabel('Time')
        plt.ylabel('Mean Square Error')
        plt.xlim(time_steps[0], time_steps[-1])
        plt.legend()
        plt.title('Mean Squared Error vs Time')
        plt.tight_layout()          
