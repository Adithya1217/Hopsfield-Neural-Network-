# Hopfield Dynamics Integration

This program executes numerical integration of continuous-time Hopfield Neural Network dynamics. Execution is deterministic. The objective is to compute the system's trajectory toward its stable fixed points.

## Operation

Input parameters dictate system initialization. 
1. `n`: Network dimensionality. 
2. `x(0)`: Initial state vector.

Integration utilizes a 4th-order Runge-Kutta (RK4) method. The system computes the strict mathematical equilibrium where dx/dt = 0 using `scipy.optimize.fsolve`.

## Modifications

Structural integrity remains uncompromised only within these explicit bounds:

- `C` (Decay vector): Alters decay velocity.
- `A` (Weight Matrix): Synaptic coupling. Nullifying the diagonal (`np.fill_diagonal(A, 0)`) enforces discrete behavior constraints.
- `I` (External Bias): External forcing function.
- `dt`: Integration time step. Lower values increase precision and elevate computational burden.
- `time_steps`: Integration duration limit.

## Execution

Dependency requirements: `numpy`, `matplotlib`, `scipy`.

Execute via `python test.py`.
