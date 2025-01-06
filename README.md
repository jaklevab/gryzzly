# Seven Years of Project Time-Tracking Data Capturing Collaboration and Failure Dynamics: The Gryzzly Dataset
This repository contains code related to the paper [Seven Years of Project Time-Tracking Data Capturing Collaboration and Failure Dynamics: The Gryzzly Dataset](https://github.com/jaklevab/gryzzly/) (currently in submission) needed to prepare the data and reproduce the analysis on project failure and its drivers.

<p float="left">
  <img src="./figures/project_task_failure.png" type="application/pdf" width="99%"/>
</p>

Project Environment Setup
-----

This project uses Python to replicate the results. Follow these steps to create and activate a **conda** environment with Python 3.8, then install the required packages from `requirements.txt`. This will also build and install any necessary Python extensions included in this package.

```bash
# 1. Create a conda environment with Python 3.8
conda create -n myenv python=3.8

# 2. Activate the newly created environment
conda activate myenv

# 3. Install the project dependencies
pip install -r requirements.txt
```

Datasets folders
----------------

This project relies upon seven datasets that should be downloaded from the accompanying [figshare](https://figshare.com/) repository and added to the ``data`` folder. It should therefore look like the following:
```
data
├── declarations.csv
├── users.csv
├── teams.csv
├── projects.csv   
├── projects_computed.csv   
├── tasks.csv   
├── tasks_computed.csv   

./
├── Figures.ipynb    # Jupyter Notebook to reproduce the paper figures
├── helper_files.py  # Helper methods for data handling
├── requirements.txt # pip requirements file
└── clean_data.sh 	 # Cleaning file used to export the figshare dataset

figures
├── circ_ccdf_hour_iet.pdf    # Figure 4
├── failure_streaks.pdf       # Figure 5      
└── forest_plot.pdf           # Figure 7    
└── network_stats.pdf         # Figure 2         
└── project_fail_overview.pdf # Figure 3              
└── project_task_failure.pdf  # Figure 6  
```

Analysis
--------
Once the datasets are compiled and the environment is compiled, you can run the notebook to reproduce the paper figures. In order to do so, launch a Jupyter notebook server and run the notebook ``Figures.ipynb``.
All computations were performed on a 3.2 GHz Apple M1 processor (8 cores, 8 GB RAM), and the notebook took approximately **27 minutes** to execute.

### Citation
If you use the code, data, or analysis results in this paper, we kindly ask that you cite the paper above as:

> _Seven Years of Project Time-Tracking Data Capturing Collaboration and Failure Dynamics: The Gryzzly Dataset_ , J. Levy Abitbol, L. Arod, 2024, .
