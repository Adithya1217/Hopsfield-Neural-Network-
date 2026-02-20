import numpy as np

def hopfield_dynamics(x, C, A, I):
    return -C * x + A * np.tanh(x) + I

def rk4_step(x, dt, C, A, I):
    k1 = hopfield_dynamics(x, C, A, I)
    k2 = hopfield_dynamics(x + 0.5 * dt * k1, C, A, I)
    k3 = hopfield_dynamics(x + 0.5 * dt * k2, C, A, I)
    k4 = hopfield_dynamics(x + dt * k3, C, A, I)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# System parameters
C, A, I = 1.0, 2.0, 0.0
dt, time_steps = 0.05, 150

# Testing a single starting state
x_current = 1.5 

# Integration loop
for step in range(time_steps):
    x_current = rk4_step(x_current, dt, C, A, I)

print(f"Final stabilized state: {x_current:.4f}")
        
    




