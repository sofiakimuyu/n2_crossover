"""
Attempt to replicate Nitrogen Concentration ObserverModel Based on the paper Anode purge management for hydrogen utilization and stack durability
improvement of PEM fuel cell systems
Not fully finished but with the intention to:
- Implement a voltage-based Luenberger-style observer
- Integrate other parameters to be able to run simulations to make more informed decisions

"""

import numpy as np
import matplotlib.pyplot as plt

# Reuse core functions from earlier demo (simplified inline)

R = 8.314
F = 96485
T_an = 358.15
P_an = 1.6e5
P_N2_ca = 0.79 * 1.2e5
V_an = 1.25e-4
b = R*T_an/(2*F)

jd_pts = np.array([0.4, 0.6])
k_pts = np.array([1.490e-9, 1.994e-9])

def kN2_from_jd(jd):
    return np.interp(jd, jd_pts, k_pts)

def alpha_dot(alpha, jd):
    k = kN2_from_jd(jd)
    return (R*T_an/V_an) * k * (P_N2_ca/P_an - alpha)

def voltage_drop(alpha, alpha_ref):
    h2_frac = max(1e-6, 1.0 - alpha)
    h2_ref = max(1e-6, 1.0 - alpha_ref)
    return b * np.log(h2_frac / h2_ref)

def observer_dot(alpha_hat, jd, y_meas, alpha_ref, H):
    k = kN2_from_jd(jd)
    model = (R*T_an/V_an) * k * (P_N2_ca/P_an - alpha_hat)
    h2_frac_hat = max(1e-6, 1.0 - alpha_hat)
    h2_ref = max(1e-6, 1.0 - alpha_ref)
    y_hat = b * np.log(h2_frac_hat / h2_ref)
    return model + H * (y_meas - y_hat)

# Simulation function for one H value
def simulate_observer(H, t_end=200, dt=0.05, jd_val=0.6, noise_std=0.0005):
    t = np.arange(0, t_end+dt, dt)
    alpha_true = np.zeros_like(t)
    alpha_true[0] = 0.02
    alpha_ref = alpha_true[0]
    alpha_hat = np.zeros_like(t)
    alpha_hat[0] = 0.02
    rng = np.random.default_rng(42)
    y_meas = np.zeros_like(t)
    for k in range(1, len(t)):
        alpha_true[k] = alpha_true[k-1] + dt*alpha_dot(alpha_true[k-1], jd_val)
        deltaV = voltage_drop(alpha_true[k], alpha_ref)
        y_meas[k] = deltaV + noise_std*rng.standard_normal()
        alpha_hat[k] = alpha_hat[k-1] + dt*observer_dot(alpha_hat[k-1], jd_val, y_meas[k], alpha_ref, H)
    return t, alpha_true, alpha_hat

# Compare different H values
H_values = [0.05, 0.3, 2.0]
results = {}
for H in H_values:
    results[H] = simulate_observer(H)

# Plot comparison
plt.figure(figsize=(9,5))
for H, (t, alpha_true, alpha_hat) in results.items():
    plt.plot(t, alpha_hat, '--', label=f'Estimate H={H}')
plt.plot(t, alpha_true, 'k', linewidth=2, label='True')
plt.xlabel("Time [s]")
plt.ylabel("Anode N2 fraction α")
plt.title("Effect of Observer Gain H on Nitrogen Estimate")
plt.legend()
plt.tight_layout()
plt.show()

# Simulation with H2 loss tracking
def simulate_with_h2_loss(H=0.3, t_end=600, dt=0.1, jd_val=0.6,
                          purge_threshold=0.05, purge_loss_fraction=0.05,
                          noise_std=0.0005):

#purge_loss_fraction: fraction of anode H2 lost during a purge (simple approximation)

    t = np.arange(0, t_end+dt, dt)
    alpha_true = np.zeros_like(t)
    alpha_true[0] = 0.02
    alpha_ref = alpha_true[0]
    alpha_hat = np.zeros_like(t)
    alpha_hat[0] = 0.02

    rng = np.random.default_rng(42)
    y_meas = np.zeros_like(t)

    # Track H2 consumed vs lost
    H2_consumed = np.zeros_like(t)
    H2_purged = np.zeros_like(t)
    H2_total_used = np.zeros_like(t)

    # Assume stack current density jd_val over area 250 cm² (from paper table)
    A_c = 250.0  # cm²
    I_stack = jd_val * A_c  # A
    mol_H2_reaction_rate = I_stack / (2*F)  # mol/s (H2 consumption by reaction)

    for k in range(1, len(t)):
        alpha_true[k] = alpha_true[k-1] + dt*alpha_dot(alpha_true[k-1], jd_val)
        deltaV = voltage_drop(alpha_true[k], alpha_ref)
        y_meas[k] = deltaV + noise_std*rng.standard_normal()
        alpha_hat[k] = alpha_hat[k-1] + dt*observer_dot(alpha_hat[k-1], jd_val, y_meas[k], alpha_ref, H)

        # Track H2 consumption
        H2_consumed[k] = H2_consumed[k-1] + mol_H2_reaction_rate*dt

        # Purge logic: if nitrogen exceeds threshold, purge
        if alpha_true[k] >= purge_threshold:
            # Hydrogen lost ~ purge_loss_fraction of anode hydrogen inventory
            P_H2_an = (1 - alpha_true[k]) * P_an
            n_total_anode = P_an*V_an/(R*T_an)
            n_H2_anode = P_H2_an*V_an/(R*T_an)
            H2_purged[k] = H2_purged[k-1] + purge_loss_fraction * n_H2_anode

            # Reset nitrogen concentration
            alpha_true[k] = 0.02
            alpha_hat[k] = min(alpha_hat[k], 0.03)
            alpha_ref = alpha_true[k]

        else:
            H2_purged[k] = H2_purged[k-1]

        H2_total_used[k] = H2_consumed[k] + H2_purged[k]

    return t, alpha_true, alpha_hat, H2_consumed, H2_purged, H2_total_used

# Run simulation
t, alpha_true, alpha_hat, H2_consumed, H2_purged, H2_total_used = simulate_with_h2_loss()

# Plot nitrogen estimate
plt.figure(figsize=(8,4))
plt.plot(t, alpha_true, label="True α_N2,an")
plt.plot(t, alpha_hat, '--', label="Estimated α_N2,an")
plt.axhline(0.25, linestyle=":", color='r', label="Purge threshold (25%)")
plt.xlabel("Time [s]")
plt.ylabel("N2 fraction in anode")
plt.title("Nitrogen Concentration and Observer")
plt.legend()
plt.tight_layout()
plt.show()

# Plot hydrogen usage
plt.figure(figsize=(8,4))
plt.plot(t, H2_consumed*1000, label="H2 consumed in stack [mmol]")
plt.plot(t, H2_purged*1000, label="H2 lost in purges [mmol]")
plt.plot(t, H2_total_used*1000, label="Total H2 required [mmol]")
plt.xlabel("Time [s]")
plt.ylabel("Hydrogen [mmol]")
plt.title("Hydrogen Consumption and Loss over Time")
plt.legend()
plt.tight_layout()
plt.show()

