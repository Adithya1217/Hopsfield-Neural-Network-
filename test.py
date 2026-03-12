import numpy as np
import matplotlib.pyplot as plt

def hopfield_dynamics(x, C, A, I):
    return -C * x + A @np.tanh(x) + I

def rk4_step(x, dt, C, A, I):
    k1 = hopfield_dynamics(x, C, A, I)
    k2 = hopfield_dynamics(x + 0.5 * dt * k1, C, A, I)
    k3 = hopfield_dynamics(x + 0.5 * dt * k2, C, A, I)
    k4 = hopfield_dynamics(x + dt * k3, C, A, I)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

n = int(input("number of neurons: "))
x_current = np.array([float(input(f"x_{i+1}(0): ")) for i in range(n)])

C = np.full(n, 2)
A = np.full((n, n), 1.5)
np.fill_diagonal(A, 1.5)
#I = np.zeros(n)
I = np.full(n, 2)
dt = 0.05
time_steps = 300

trajectory = []
times = []

for step in range(time_steps):
    x_current = rk4_step(x_current, dt, C, A, I)
    trajectory.append(x_current.copy())
    times.append(step * dt)

trajectory = np.array(trajectory)

print(f"Final stabilized state: {x_current}")
print(C)
print(A)
print(I)
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
plt.figure(figsize=(10, 5))
for i in range(n):
    plt.plot(times, trajectory[:, i], marker='o', markersize=2,
             color=colors[i % len(colors)], label=f"x{i+1}(t)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time')
plt.ylabel('State x')
plt.title('Hopfield Dynamics Trajectory')
plt.legend()
plt.show()
def eq(X, C, A, I):
    # This equation represents dx/dt = 0
    # We want to find X such that the derivative is 0
    return -C * X + A @ np.tanh(X) + I

# You can solve this equation to find the equilibrium points using scipy:
from scipy.optimize import fsolve

# Use the final simulated state as an initial guess for the solver
#equilibrium_point = fsolve(eq, x_current, args=(C, A, I))
#print(f"Solved equilibrium point (fixed point): {equilibrium_point}")