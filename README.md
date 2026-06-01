**SimBatch: Batch Simulation Toolbox for Reservoir Engineers**

SimBatch is a Python-based batch simulation and evaluation toolkit designed for reservoir engineers and researchers working with simulators like RFD tNavigator and SLB Eclipse. It enables automated, scalable execution of multiple simulation cases and extracts key performance indicators for comparative analysis and optimisation.

Key Features:

Automated batch execution of multiple .DATA files on CPU or GPU

Integration-ready with tNavigator and Eclipse via command-line interface

Smart output parsing of .RSM files for selected keywords and time series

Report generation of end-time and time-series outputs in clean CSV format

Customisable decision variables via VARIABLES.INC with bounds and units

Parallel execution with user-defined concurrency (optional)

Technologies Used:

Python 3, NumPy, Pandas, concurrent.futures, subprocess

Compatible with RFD tNavigator and SLB Eclipse .DATA deck structure

Use Cases:
Sensitivity analysis
Scenario comparison
History matching & optimisation\

**Author: Amirsaman Rezaeyan**

---
Cite as:\
Amirsaman Rezaeyan, 2024, SimBatch: a Batch Simulation Running Tool in Python with Evaluations Through Reservoir Simulators. https://github.com/EarthSciTech/SimBatch
