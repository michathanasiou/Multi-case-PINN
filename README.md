# On the Performance of Physics-Informed Neural Networks for Hemodynamic Predictions in Parametrized Vascular Stenoses
This repository contains the code accompanying the paper:
“On the Performance of Physics-Informed Neural Networks for Hemodynamic Predictions in Parametrized Vascular Stenoses.”

• Multi-case.py —> trains a multi-case PINN across multiple vascular geometries and flow conditions, employing an adaptive SiLU activation function.

• Single-case.py —> trains a single-case PINN on an individual stenotic configuration.

Both implementations utilize Physics-Informed Neural Networks (PINNs) trained without labeled data, relying solely on the parameterized incompressible steady-state continuity and Navier–Stokes equations to model blood flow in simplified 2D stenotic geometries.



