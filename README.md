# Dynamic Mode Decomposition
*DMD* is an algorithm for dimensionality reduction and stability analysis. It allows to extract the dominant patterns and structures of a dynamical system, in order to 
reduce the computational effort, beside getting informations about its development. It is based on the *Singular Value Decomposition* (*SVD*), a very powerful algebraic
result thanks to which rectangular matrices can be decomposed. In addition, *SVD* is strictly related to the *Principal Component Analysis* (*PCA*), that is a benchmark
for dimensionality reduction in many data analysis applications and represents a useful alternative to construct the *DMD*
(click [here](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca) for a discussion on *PCA* and
*SVD* relation).

## Algorithm
Consider a dynamical system and a series of snapshots $X = [x_1, x_2, ..., x_{n+1}]$ in time of a certain evolving quantity, with $x_k \in \mathbb{R}^m$. These snapshots
can be arranged in a $m \times n$ matrix $X$, where we neglect the last column. We can then define a second data-matrix $X'$, which coincides with $X$ but it is shifted
by one time-step into the future. We are looking for the best linear operator $A$ such that:

$$  
X' = AX
$$

Matrix $X$ columns represent "screenshots" of the system at issue as it evolves, while rows could represent points in space. In a fluid dynamics context, $X_{ij}$ would be
for example the velocity of the $i^{th}$ point of the domain at the $j^{th}$ time-step. Due to this reason, $X$ is often a tall and skinny matrix, since there are usually
few snapshots and many points at disposal. We can then apply the $SVD$ to matrix $X$, obtaining:

$U$ and $V$ are square orthogonal matrices and their columns constitute two possible orthonormal bases to represent data; $\Sigma$ is diagonal and it contains the
so-called **singular values** on its entries. Singular values represent the importance of each left singular vector (columns of $U$), namely how much each one contributes
to construct data. Since in many dynamical systems there are often coherent structures which just repeat themselves equally, we can decide to neglect a certain number
of the lowest singular values (ergo, left singular vectors) based on data. These discarded elements are associated to linearly dependent vectors which provide no useful
informations, and they often guarantee a huge dimensionality reduction (from $10^2-10^6$ to some tens) by taking away just a few percentage of the total contribution. 

Matrix $A$ can then be written as

$$
A = X'X^+ = X'V\Sigma^{-1}U^t
$$

We can find the reduced matrix $\tilde{A}$ by projecting $A$ on the principal vectors of the reduced matrix $U_r$:

$$
\tilde{A} = U_r^t A U_r = U_r^t X' V_r \Sigma_r^{-1}
$$

Thanks to the presence of the coherent structures that we used to truncate the rank, $A$ and $\tilde{A}$ are similar, so they share the same non-zero eigenvalues.
By solving the eigenvalue problem for matrix $\tilde{A}$, we can then find the eigenvectors $W$ and use them to recover the eigenvectors $\Phi$ of the original
operator $A$. The latter can be found as:

$$
\Phi = X' V_r \Sigma_r^{-1} W
$$

Columns of $\Phi$ represent the so-called **DMD modes**, namely the dominant structures of the system which can be studied to get informations about the motion
development (e.g., the absolute value of eigenvalues tells if the corresponding mode will shrink, be stable or diverge) and reconstruct data. 
The generic formula that allows to retrieve the state vector $x_n$ is

$$
x_n = \Phi \Lambda^n b
$$

where $\Lambda$ is the matrix of eigenvalues and 

$$
b = \Phi^{-1} x_0
$$

## Repository structure
The present repository contains the following files and folders:
- *.gitignore*
- *README.md*
- *LICENSE*
- *requirements.txt*
- *DMD* folder -> contains the source code of the Dynamic Mode Decomposition algorithm, divided into:
    - *config.py* -> contains constant variables
    - *data_loader.py* -> code section responsible for the loading of data that will be used
    - *data_processor.py* -> code section responsible for the processing of loaded data
    - *functions.py* -> contains the (only) function used in the simulation module
    - *plotter.py* -> contains the class **Plotter**, whose methods are used for data and results visualization
    - *simulation.py* -> the DMD algorithm itself

- *tests* folder -> contains the different tests for the various modules of the code. Each file name refers to the specific module tested:
    - *test_data_loader.py*
    - *test_data_processor.py*
    - *test_functions.py*
    - *test_plotter.py*
    - *test_simulation.py*

# Data
The present project has been realized through the application of the *DMD* algorithm to a simulated fluid dynamics dataset. Data belongs to a Python library 
called *flowTorch*, whose documentation can be found [here](https://github.com/FlowModelingControl/flowtorch).
Two folders containing datasets of different phenomena can be downloaded, the [full one](https://cloud.tu-braunschweig.de/s/sJYEfzFG7yDg3QT) ($\approx 2.6GB$)
or the [reduced one](https://cloud.tu-braunschweig.de/s/b9xJ7XSHMbdKwxH) ($\approx 411MB$). Both can be properly used to implement the algorithm, the unique difference
is given by the number of snapshots included for each dataset. Specifically, the one of interest is called *of_cylinder2D_binary*. It contains 401 snapshots of
different vector/scalar fields (pressure, vorticity, surface flux and velocity) of a fluid past a 2D cylinder. Code takes into account the vorticity field.

*Disclaimer*: the present repository, including functions, testing and the algorithm itself works only for the vorticity field. 

Furthermore, code is mainly meant to show a panorama of the Dynamic Mode Decomposition algorithm, specifically for its implementation and results visualization. For this reason, there's no much freedom on code flow, such as input values that user can insert are not present. This is due to a prior study and analysis of data to ensure the best results in terms of efficiency (e.g. the choice of 99.5% as the threshold value for SVD truncation allows to cut data as much as possible with the minimum loss of information. See *DMD/simulation.py*); anyway, user can decide to modify variables arbitrarily to explore different scenarios.

## Getting started
In a prompt, navigate in an arbitrary directory and clone the repository by running the following command:
```git
git clone https://github.com/lelesqc/DynamicModeDecomposition
```
Navigate into the project directory
```git
cd DynamicModeDecomposition
```
and install requirements by running:
```git
pip install -r requirements.txt
```
This command will automatically download *flowTorch* from its original repository, ensuring the complete download of the latest version of the library. This is because the *PyPI* repository of Python libraries isn't updated to the most recent version.
Alternatively, *flowTorch* can be downloaded manually by running
```git
pip3 install git+https://github.com/FlowModelingControl/flowtorch
# or install a specific branch, e.g., aweiner
pip3 install git+https://github.com/FlowModelingControl/flowtorch.git@aweiner
```
For further information, user can visit this [link](https://github.com/FlowModelingControl/flowtorch).

Now, if not done yet, download Jupyter Notebook (to run the main code, which is a *.ipynb* file) through:
```git
pip install jupyter
```
Before proceeding, data have to be downloaded as well.

Choose one of the two available datasets (full or reduced), download it from the link above and place it in an arbitrary directory. The downloaded file is a compressed archive with extension *.tar.gz*. Now, to extract the archive run the following commands:

### Linux
```git
# full dataset
tar xzf datasets_13_09_2022.tar.gz
# reduced dataset
tar xzf datasets_minimal_13_09_2022.tar.gz
```
To tell *flowTorch* where the datasets are located, define the *FLOWTORCH_DATASETS* environment variable:

```git
# add export statement to bashrc; assumes that the extracted 'datasets' or 'datasets_minimal'
# folder is located in the current directory
# full dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets/\"" >> ~/.bashrc
# reduced dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets_minimal/\"" >> ~/.bashrc
# reload bashrc
. ~/.bashrc
```
### Windows 10/11
```git
# full dataset
tar -xf datasets_13_09_2022.tar.gz
# reduced dataset
tar -xf datasets_minimal_13_09_2022.tar.gz
```
To tell *flowTorch* where the datasets are located, define the *FLOWTORCH_DATASETS* environment variable:

```git
setx FLOWTORCH_DATASETS "C:\path\to\folder\of\datasets"
```
To verify that the environment variable has been correctly defined, restart the prompt and run:

```git
echo %FLOWTORCH_DATASETS%
```
This command should print the path to the datasets folder.

### Run the simulation
If all the previous steps have been completed, user can already see the notebook outputs by opening it or re-run the simulation and see results.
While being on the project directory, run:
```git
jupyter notebook
```
and open *DMD\main.ipynb* to see the notebook.

### Access to data in Python
If user wants to access data on its own, let's see how to use them in Python:

```python
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box

path = DATASETS["of_cylinder2D_binary"]
loader = FOAMDataloader(path)

t_steps = loader.write_times
fields = loader.field_names
pts = loader.vertices[:, :2]    # Discard z-coordinate since simulation is only 2D

# Vortex shedding is complete after 4s, keep only time steps between 4s and 10s
times = [t for t in t_steps if float(t) >= 4.0] 

for t in times:
    snapshots = loader.load_snapshot("vorticity", t)
```
In this way data of interest can be retrieved from the dataset.

## Implementation of an easier version
The code provided in our example manually perform the algorithm, by translating the theoretical approach we have seen above into code. However, *SVD* and *DMD*
algorithm itself are well-known topics and libraries which do the whole job are already existing. Some examples of these modules are
- [flowTorch.analysis](https://flowmodelingcontrol.github.io/flowtorch-docs/1.0/flowtorch.analysis.html)
- [PyDMD](https://pydmd.github.io/PyDMD/code.html)

Let's see how easily can *SVD* and *DMD* algorithms be implemented through *flowTorch*:

```python
import numpy as np
import torch as pt
from functions import find_optimal_rank
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD, DMD

# -------------- CONFIGURATION -----------------

path = DATASETS["of_cylinder2D_binary"]
loader = FOAMDataloader(path)

t_steps = loader.write_times
fields = loader.field_names
pts = loader.vertices[:, :2]    # Discard z-coordinate since simulation is only 2D

# Vortex shedding is complete after 4s, keep only time steps between 4s and 10s
times = [t for t in t_steps if float(t) >= 4]
dt = round(float(times[1]) - float(times[0]), 3)

mask = mask_box(pts, lower = [.1, -1], upper = [.75, 1])    # Boolean values matrix, 1 to keep, 0 to discard
data_matrix = pt.zeros((pt.count_nonzero(mask), len(time_steps)), dtype = pt.float32)

for i, t in enumerate(times):
    snapshots = loader.load_snapshot("vorticity", t)
    data_matrix[:, i] = pt.masked_select(snapshots[:, 2], mask)

# -------------- DMD -----------------

rank_datamatrix = min(data_matrix[:, :-1].size())
svd = SVD(data_matrix[:, :-1], rank=rank_datamatrix)    # Reduced SVD on matrix X = data_matrix without last column
singular_vals = svd.s

thr = 99.5
optimal_rank = find_optimal_rank(singular_vals, thr)
dmd = DMD(data_matrix, dt = dt, rank=optimal_rank)


"""
MAIN PROPERTIES of flowtorch.analysis.SVD:
- U    # Matrix of left singular vectors
- V    # Matrix of right singular vectors
- s    # Singular values
- opt_rank    # Automatically computed optimal rank (didn't make use of it)

MAIN PROPERTIES of flowtorch.analysis.DMD:
- amplitude    # Amplitude of DMD modes
- dynamics    # Time dynamics
- eigvals 
- eigvecs
- frequency    # Frequency of DMD modes
- modes    # DMD modes
- reconstruction    # Data matrix re-obtained starting from DMD modes

"""
```

These functions and properties of *flowtorch.analysis* are used in our testing module to verify results. This example is constructed through a unique implementation code
(plotting of data and results excluded), while in *DMD* folder the different parts of the code (configuration, data loader etc.) are displayed individually.
