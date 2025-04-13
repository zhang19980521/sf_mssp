# This code is used for numerical simulation of partially reproduced paper
# "Spectrum forming and its high-performance implementations for the blade
# tip timing signal processing" (Submitted to Mechanical Systems and Signal
# Processing, haven't been peer reviewed)
# author: Chenyu Zhang
# lastupdate: April 13th, 2025
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from numpy.linalg import norm


def clean_fc(Psi, steer_vecs, f_range, f_interval, kappa, maxIter):
    """
    This function is used to obtain the output of CLEAN-FC
    author: Chenyu Zhang
    lastupdate: April 13th, 2025
    out_clean_fc = clean_fc(Psi, steer_vecs, f_range, f_interval, kappa, maxIter)
        Psi: cross-correlation matrix
        steer_vecs: steering vectors for each frequency
        f_range: frequency range ([min, max])
        f_interval: frequency interval
        kappa: safety factor in Eq. (24)
        maxIter: max iteration number
    """
    freq_scan = np.arange(f_range[0], f_range[1] + f_interval, f_interval)
    N_f = len(freq_scan)
    # Start CLEAN-FC
    Clean_spectrum = np.zeros(N_f)
    out_clean_fc = np.zeros(N_f)
    Degraded_Psi = Psi.copy()

    Dirty_Spectrum = np.zeros(N_f)
    for ii in range(N_f):
        Dirty_Spectrum[ii] = np.abs(steer_vecs[:, ii].conj().T @ Degraded_Psi @ steer_vecs[:, ii]) / (
                    np.linalg.norm(steer_vecs[:, ii], 2) ** 4)  # See Eq. (22)

    Psi_curr = np.sum(np.abs(Degraded_Psi.flatten()))
    count = 0
    Psi_prev = 1e10

    while (Psi_curr < Psi_prev) and (count < maxIter):
        # Determine the peak frequency in the spectrum
        Spec_max = np.max(np.abs(Dirty_Spectrum))  # Search the maximum value
        index_max = np.argmax(np.abs(Dirty_Spectrum))  # Obtain its index
        Spectrum_maxClean = np.zeros_like(Clean_spectrum)
        Spectrum_maxClean[index_max] = 1
        ispositiv = np.abs(Dirty_Spectrum[index_max]) > 0  # Delete negative spectrum values

        hmax = Degraded_Psi @ steer_vecs[:, index_max] / (Spec_max * (np.linalg.norm(steer_vecs[:, index_max],
                                                                                     2) ** 2))  # steering vector of the frequency related to the maximum value

        # Update degraded Psi
        Cource = Spec_max * np.outer(hmax, hmax.conj())  # Contribution of the frequency related to the maximum value
        Degraded_Psi = Degraded_Psi - kappa * Cource  # see Eq. (24)

        # Update dirty spectrum
        for ii in range(N_f):
            Dirty_Spectrum[ii] = np.abs(steer_vecs[:, ii].conj().T @ Degraded_Psi @ steer_vecs[:, ii] / (
                        np.linalg.norm(steer_vecs[:, ii], 2) ** 4))

        # Update clean spectrum
        Clean_spectrum = Clean_spectrum + kappa * Spec_max * Spectrum_maxClean * ispositiv

        # Convergence checking
        Psi_prev = Psi_curr
        Psi_curr = np.sum(np.abs(Degraded_Psi.flatten()))
        count += 1

    # Delete negative spectrum power
    Dirty_Spectrum[np.abs(Dirty_Spectrum) < 0] = 0

    # Final spectrum equal to the sum of clean spectrum and dirty spectrum
    CLEAN_FC_spectrum = Clean_spectrum + Dirty_Spectrum

    # Accumulate the frequency components
    out_clean_fc = CLEAN_FC_spectrum
    return out_clean_fc


def soft(w, delta):
    """
    Soft threshold operator in Eq. (21)
    author: Chenyu Zhang
    lastupdate: April 13th, 2025
    out = soft(w, delta)
        w: a column vector
        delta: threshold
    """
    out = np.maximum(np.abs(w) - delta, 0)
    out[out < 0] = 0  # Remove negative values
    return out


def de_gmcp(A, out_sf, gamma, eta, epsilon, R):
    """
    This function is used to obtain the output of De-GMCP (Algorithm 1)
    author: Chenyu Zhang
    lastupdate: April 13th, 2025
    out_gmcp = de_gmcp(A, out_sf, gamma, eta, epsilon, R)
        A: FPSF matrix in Eq. (13)
        out_sf: Sf output (dirty spectrum)
        gamma: within 0~1, control the convexity of GMCP
        eta: regularization parameter
        epsilon: convergence threshold
        R: convergence threshold
    """
    # Parameter initialization
    rho = max(1, gamma / (1 - gamma)) * np.linalg.norm(A.conj().T @ A)
    mu = 3 / (2 * rho)
    _, n = A.shape  # Size of FPSF matrix
    out_gmcp_temp = np.zeros((n, R))
    v = np.zeros(n)
    e = np.ones(n)
    B = np.sqrt(gamma / eta) * A

    # Start iteration
    for ii in range(1, R):
        w = out_gmcp_temp[:, ii - 1] - mu * A.conj().T @ (A @ out_gmcp_temp[:, ii - 1] - out_sf) + eta * mu * B.conj().T @ B @ (
                    v - out_gmcp_temp[:, ii - 1])
        u = v - mu * eta * B.conj().T @ B @ (v - out_gmcp_temp[:, ii - 1])
        out_gmcp_temp[:, ii] = soft(w, eta * mu)
        v = soft(u, eta * mu)

        residual = np.linalg.norm(out_gmcp_temp[:, ii] - out_gmcp_temp[:, ii - 1]) / np.linalg.norm(
            out_gmcp_temp[:, ii])
        if residual < epsilon:  # Convergence check
            break

    out_gmcp = out_gmcp_temp[:, -1]
    return out_gmcp


def fsf(Psi, steer_vecs, chi):
    """
    This function is used to obtain the output of functional spectrum forming
    author: Chenyu Zhang
    lastupdate: April 13th, 2025
    out_fsf = fsf(Psi, steering_vecs, chi)
        Psi: cross-correlation matrix
        steering_vecs: steering vectors for each frequency
        chi: exponent for functional spectrum forming (chi>1)
    """
    # SVD for cross-correlation matrix (see Eq. (25))
    U, S, Vh = svd(Psi)
    S_chi = np.diag(S ** (1 / chi))
    Psi_chi = U @ S_chi @ Vh  # see Eq. (26)

    out_fsf = np.zeros(steer_vecs.shape[1])
    for ii in range(steer_vecs.shape[1]):
        numerator = np.abs(steer_vecs[:, ii].conj().T @ Psi_chi @ steer_vecs[:, ii])
        denominator = (np.linalg.norm(steer_vecs[:, ii], 2) ** 4)
        out_fsf[ii] = (numerator / denominator) ** chi  # see Eq. (27)

    return out_fsf


def sf(Psi, steer_vecs):
    """
    This function is used to obtain the output of spectrum forming (dirty map)
    author: Chenyu Zhang
    lastupdate: April 13th, 2025
    out_sf = sf(Psi, steer_vecs)
        Psi: cross-correlation matrix
        steering_vecs: steering vectors for each frequency
    """
    n_f = steer_vecs.shape[1]  # Number of frequencies to be formed
    out_sf = np.zeros(n_f)
    for ii in range(n_f):
        out_sf[ii] = np.abs(steer_vecs[:, ii].conj().T @ Psi @ steer_vecs[:, ii]) / (
                    np.linalg.norm(steer_vecs[:, ii], 2) ** 4)  # see Eq. (9)
    return out_sf

# Main
# Disc rotation
fr = 940 / 60  # Rotational frequency

# Set the signal
f = np.array([45, 220])  # Signal frequency
amp = np.array([0.1, 0.2])  # Amplitude
phase = np.array([np.pi / 6, np.pi / 5])  # Phase
n_rev = 500  # Number of revolutions simulated

# Set the TOA probes
delta = np.array([16, 159, 177, 181, 240, 276])  # Azimuth angle of TOA probes (unit: °)
n_p = len(delta)  # Number of TOA probes

# Signal sampling
t = np.zeros(n_p * n_rev)  # Time stamp initialization
for ii in range(n_rev):
    for jj in range(n_p):
        t[n_p * ii + jj] = (1 / fr) * (ii + delta[jj] / 360)

d = np.zeros(len(t))  # Blade displacement initialization
for ii in range(len(f)):
    d_temp = amp[ii] * np.sin(2 * np.pi * f[ii] * t + phase[ii])
    d = d + d_temp

# B-Hankel matrix construction
l_s = 30 * n_p  # Length of snapshot
n_s = l_s  # Number of snapshots (specially, l_s = n_s in this simulation)
D = np.zeros((l_s, n_s), dtype=complex)  # B-Hankle matrix initialization (see Eq. (5))
for ii in range(n_s):
    start_idx = ii * n_p
    end_idx = start_idx + l_s
    if end_idx > len(d):
        end_idx = len(d)
    D[:, ii] = d[start_idx:end_idx]

Psi = D @ D.conj().T / n_s  # Cross-correlation matrix in Eq. (6)

# Spectrum forming to obtain aliasing spectrum
f_range = np.arange(-300, 301, 1)  # Frequency range with interval 1 Hz
transfer_matrix = np.zeros((l_s, len(f_range)), dtype=complex)  # Transfer matrix initialization (H in Eq. (5))
steer_vecs = np.zeros((l_s, len(f_range)), dtype=complex)  # Steering vectors initialization (h in Eqs. (8) and (9))
for ii in range(len(f_range)):
    transfer_matrix[:, ii] = np.exp(1j * 2 * np.pi * f_range[ii] * t[:l_s])
    steer_vecs[:, ii] = np.conj(np.exp(1j * 2 * np.pi * f_range[ii] * t[:l_s]))

# Calculate the frequency point spread function (FPSF)
A = np.zeros((len(f_range), len(f_range)))  # Initialization of the FPSF matrix
for ii in range(len(f_range)):
    for jj in range(len(f_range)):
        # Ensure steer_vecs and transfer_matrix are 2D arrays
        steer_vec = np.reshape(steer_vecs[:, ii], (-1, 1))
        transfer_vec = np.reshape(transfer_matrix[:, jj], (-1, 1))

        # Compute the contribution
        contribution = np.abs(steer_vec.conj().T @ transfer_vec @ transfer_vec.conj().T @ steer_vec)
        contribution /= (np.linalg.norm(steer_vec, 2) ** 2 * np.linalg.norm(transfer_vec, 2) ** 2)

        A[ii, jj] = contribution  # see Eq. (14)
# Output of different methods
# Spectrum forming (SF) in Section 3
out_sf = sf(Psi, steer_vecs)

# Deconvolution with NNLS regression (De-NNLS) in Section 4.2.1
from scipy.optimize import nnls

out_nnls = nnls(np.abs(A), np.abs(out_sf))[0]  # Deconvolution with NNLS (De-NNLS) in Section 4.2.1

# Deconvolution with GMCP (De-GMCP) in Section 4.2.2
gamma = 0.3  # Within 0~1, control the convexity of GMCP
eta = 5e-4  # Regularization parameter
epsilon = 1e-4  # Convergence threshold
R = 1000  # Max iteration number
out_gmcp = de_gmcp(np.abs(A), np.abs(out_sf), gamma, eta, epsilon, R)

# CLEAN based on frequency coherence (CLEAN-FC) in Section 4.2.3
f_range_clean = [f_range[0], f_range[-1]]  # Frequency range ([min, max])
f_interval = 1  # Frequency interval
kappa = 0.5  # Safety factor in Eq. (24)
maxIter = 100  # Max iteration number
out_clean_fc = clean_fc(Psi, steer_vecs, f_range_clean, f_interval, kappa, maxIter)

# Functional spectrum forming (FSF) in Section 4.3
chi = 100  # Exponent for functional spectrum forming (chi>1)
out_fsf = fsf(Psi, steer_vecs, chi)

# Transfer the double-sided spectrum to single-sided spectrum
f_index_middle = (len(f_range) - 1) // 2  # Index of 0 Hz

# SF single-sided spectrum
out_sf_single = np.zeros(f_index_middle+1)
out_sf_single[0] = np.sqrt(np.abs(out_sf[f_index_middle]))
out_sf_single[1:] = np.flip(np.sqrt(np.abs(out_sf[0:f_index_middle]))) + np.sqrt(np.abs(out_sf[f_index_middle + 1:]))

# De-NNLS single-sided spectrum
out_nnls_single = np.zeros(f_index_middle+1)
out_nnls_single[0] = np.sqrt(np.abs(out_nnls[f_index_middle]))
out_nnls_single[1:] = np.flip(np.sqrt(np.abs(out_nnls[0:f_index_middle]))) + np.sqrt(np.abs(out_nnls[f_index_middle + 1:]))

# De-GMCP single-sided spectrum
out_gmcp_single = np.zeros(f_index_middle+1)
out_gmcp_single[0] = np.sqrt(np.abs(out_gmcp[f_index_middle]))
out_gmcp_single[1:] = np.flip(np.sqrt(np.abs(out_gmcp[0:f_index_middle]))) + np.sqrt(np.abs(out_gmcp[f_index_middle + 1:]))

# CLEAN-FC single-sided spectrum
out_clean_single = np.zeros(f_index_middle+1)
out_clean_single[0] = np.sqrt(np.abs(out_clean_fc[f_index_middle]))
out_clean_single[1:] = np.flip(np.sqrt(np.abs(out_clean_fc[0:f_index_middle]))) + np.sqrt(np.abs(out_clean_fc[f_index_middle + 1:]))

# FSF single-sided spectrum
out_fsf_single = np.zeros(f_index_middle+1)
out_fsf_single[0] = np.sqrt(np.abs(out_fsf[f_index_middle]))
out_fsf_single[1:] = np.flip(np.sqrt(np.abs(out_fsf[0:f_index_middle]))) + np.sqrt(np.abs(out_fsf[f_index_middle + 1:]))

# Spectrum visualization of different methods
f_single_side = f_range[f_index_middle:]
plt.figure(figsize=(10, 15))

plt.subplot(5, 1, 1)
plt.plot(f_single_side, out_sf_single)
plt.title('SF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (mm)')

plt.subplot(5, 1, 2)
plt.plot(f_single_side, out_nnls_single)
plt.title('De-NNLS')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (mm)')

plt.subplot(5, 1, 3)
plt.plot(f_single_side, out_gmcp_single)
plt.title('De-GMCP')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (mm)')

plt.subplot(5, 1, 4)
plt.plot(f_single_side, out_clean_single)
plt.title('CLEAN-FC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (mm)')

plt.subplot(5, 1, 5)
plt.plot(f_single_side, out_fsf_single)
plt.title('FSF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (mm)')

plt.tight_layout()
plt.show()