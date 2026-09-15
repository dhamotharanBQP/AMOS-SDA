"""Generalized Kepler/J2 residual PINN library used by training and inference."""

import math

import numpy as np
import torch
from torch import nn

USE_DRAG = False
USE_SRP  = False
USE_SUN_AND_MOON = False
USE_J2=False

M_SC    = 220.0
A_CROSS = 1.0
CD      = 2.2
EPSILON = 0.3

SUN_POS_ECI  = np.array([1.496e11, 0.0, 0.0], dtype=float)
MOON_POS_ECI = np.array([3.84e8,   0.0, 0.0], dtype=float)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
print(f"[Info] device={DEVICE} | dtype={DTYPE}")

G         = 6.67430e-11
M_sun     = 1.989e30
M_moon    = 7.34767309e22
P_sun     = 4.57e-6
MU_EARTH  = 3.986004418e14
R_EARTH   = 6378137.0
J2        = 1.08262668e-3

# ---------------------------------------------------------------------------
# ORBITAL REGIME
#
# T0 = R0/V0 is computed per satellite, so the model is regime-agnostic in its
# scaling. These two constants only affect (a) the LEGACY gate, which divides
# by a fixed T_SCALED_MAX, and (b) the V0 input normalisation. Both are
# cosmetic under GATE_KIND='tanh', which keys off each satellite's own orbital
# period, but they still matter for the legacy path and for input conditioning.
#
# Reference T0 values: LEO ~907 s, MEO ~4900 s, GEO ~13712 s.
REGIME = "LEO"

_REGIME_TABLE = {
    #          T_SCALED_MAX   V0_SCALE (m/s)   nominal T0 (s)
    "LEO":  {"t_scaled_max": 8.8,  "v0_scale": 7500.0, "t0_ref": 907.0},
    "MEO":  {"t_scaled_max": 1.5,  "v0_scale": 3870.0, "t0_ref": 4900.0},
    "GEO":  {"t_scaled_max": 0.73, "v0_scale": 3100.0, "t0_ref": 13712.0},
}

T_SCALED_MAX = _REGIME_TABLE[REGIME]["t_scaled_max"]
V0_SCALE     = _REGIME_TABLE[REGIME]["v0_scale"]


def set_regime(name, window_duration_sec=None):
    """Select LEO / MEO / GEO and update the dependent constants.

    If window_duration_sec is given, T_SCALED_MAX is recomputed as
    window / t0_ref for that regime rather than taken from the table, which is
    what the legacy gate actually wants.
    """
    global REGIME, T_SCALED_MAX, V0_SCALE
    name = str(name).upper()
    if name not in _REGIME_TABLE:
        raise ValueError(f"REGIME must be one of {sorted(_REGIME_TABLE)}, got {name!r}")
    REGIME = name
    entry = _REGIME_TABLE[name]
    V0_SCALE = entry["v0_scale"]
    if window_duration_sec is not None:
        T_SCALED_MAX = round(float(window_duration_sec) / entry["t0_ref"], 1)
    else:
        T_SCALED_MAX = entry["t_scaled_max"]
    return {"regime": REGIME, "T_SCALED_MAX": T_SCALED_MAX, "V0_SCALE": V0_SCALE}
KEPLER_DTYPE = torch.float64

USE_J2_BASELINE = False

USE_LAT_FEATURES = False

USE_OSC2MEAN = False

GATE_POWER = 3.0

GATE_KIND    = 'legacy'
GATE_TAU_DIV = 3.0

USE_ORBIT_FEATURES = False
_N_ORBIT_FEATURES  = 5

# When False, retain only the three static orbit descriptors
# [cos(i), sin^2(i), eccentricity]. This supports experiments where orbital
# phase is represented by the Fourier time block instead of sin(ku)/cos(ku).
USE_ORBIT_PHASE_FEATURES = True

# Number of harmonics of the argument of latitude u supplied to the network.
# N=1 gives [sin u, cos u] only (orbit block = 5, total input dim 14).
# Each extra harmonic adds [sin ku, cos ku], so orbit block = 3 + 2N and
# total input dim = 9 + 3 + 2N.
#
#   N=1 -> dim 14   N=2 -> dim 16   N=3 -> dim 18   N=4 -> dim 20
#
# Measured 2026-08-08: least-squares fit of the residual (truth minus osc2mean
# baseline) onto {t, 1, sin ku, cos ku for k=1..N}, 8 satellites, 20 to 97 deg
# inclination, full 80000 s window:
#
#     N   residual   removed   marginal
#     1     2.854 km   72.8%     72.8%
#     2     0.899 km   91.4%     68.5%
#     3     0.174 km   98.3%     80.7%     <- ladder terminates here
#     4     0.173 km   98.3%      0.1%     <- nothing left
#
# The basis is complete at N=3. N=4 is measurably worthless, so do not go past
# three: it costs two inputs and buys 0.1 percent.
#
# Default is 1, which reproduces the previous behaviour exactly so existing
# checkpoints still load. Requires USE_ORBIT_FEATURES=True.
ORBIT_N_HARMONICS = 1

# ---------------------------------------------------------------------------
# Physics (PDE) branch configuration.
#
# READ THIS BEFORE ENABLING W_PDE.
#
# Measured on 2026-08-08 against DOP853 J2 truth:
#
#   1. USE_J2 defaults to False and train_opti.py never set it, so every
#      historical physics run used a PURE TWO-BODY target. Loss at the true
#      trajectory was 5.80e-07 versus 1.86e-06 at the bare baseline: only a
#      3.2x usable range, i.e. 97 percent irreducible floor. With USE_J2=True
#      the floor drops to 2.04e-11 and the range becomes 89000x.
#
#   2. Whether it helps beyond that DEPENDS ON THE FEATURE SET. Cosine between
#      the physics gradient and the HELD-OUT data gradient:
#
#      dim 14 (harmonics=1, no latitude):
#            stage   shipped   USE_J2=1   +|a_J2| norm   +gate weight
#             150      -0.05     -0.08        -0.17         -0.18
#             400      -0.34     -0.39        -0.43         -0.44
#             900      -0.29     -0.36        -0.44         -0.44
#
#      dim 20 (harmonics=3, latitude on):
#            stage   |a_total|   |a_J2|   J2-projected   lat-weighted
#             200       +0.23     +0.18       -0.25          +0.30
#             600       +0.11     +0.11       -0.16          +0.09
#            1200       +0.02     +0.03       -0.16          -0.09
#
#      So the earlier "physics always fights" result was specific to the narrow
#      feature set. With harmonics=3 the full-residual variants are weakly
#      POSITIVE early and decay to orthogonal. Physics is then a mild early
#      regulariser rather than a liability, which makes a small W_PDE worth one
#      run rather than an automatic no.
#
#      The J2-PROJECTED variant (penalising only the residual component along
#      the J2 direction, the natural reading of "let z/r drive the physics")
#      is negative at EVERY stage and is the worst of the four. It is not
#      implemented here on purpose.
#
# Recommendation: with harmonics >= 2, a small W_PDE is worth ONE controlled
# run against a no-physics control. Choose it from the measured gradient ratio
# rather than by guessing:
#
#     W_PDE = f / (|g_pde| / |g_data|)      with f = 0.1
#
# Both quantities are printed by pde_gradient_check.py against your real
# checkpoint. Run that first: it measures YOUR operating point, not a proxy.
# ---------------------------------------------------------------------------

# 'total' reproduces the shipped behaviour: divide by |a_true| + |a_nn|.
# 'j2' divides by the J2 acceleration magnitude alone. REQUIRES USE_J2=True;
# with USE_J2=False the J2 reference is identically zero, the clamp fires and
# the loss diverges to ~1e20.
PDE_NORM = 'total'

# Weight the PDE residual by the gate, so physics is only imposed where the
# network has authority to satisfy it.
PDE_GATE_WEIGHTED = False

# Residual-PDE formulation ('orbital physics minus the Kepler constraints').
#
# The default full-trajectory form already IS the residual form. Since
#     r_pred = r_base + gate*R0*N   =>   r_pred'' = r_base'' + (gate*R0*N)''
# and a_pde_nn is obtained by autograd on the FULL prediction, r_base'' is
# already inside the loss exactly. Writing it as
#     (gate*R0*N)'' - (a_grav(r_pred) - r_base'')
# is the same expression rearranged.
#
# Setting this True instead subtracts a_kepler(r_base), the two-body
# acceleration evaluated on the baseline, in place of the baseline's true
# second derivative. Those differ: the baseline carries secular J2 and the
# osc-to-mean correction, so it does not follow pure Kepler motion, and the
# difference stays in the loss.
#
# Measured context: the baseline's own pointwise ODE violation averages
# 22.5 mm/s^2 while the whole J2 perturbation is 11.4 mm/s^2, a ratio of about
# 2 (range 1.0 to 3.1 across satellites). So the term this option manipulates
# is the same size as the physics being learned, which is why it is worth
# measuring rather than assuming.
#
# Expectation: slightly worse than the default. Provided so that can be tested
# rather than argued.
PDE_SUBTRACT_KEPLER = False

_N_LAT_FEATURES = 2

# ---------------------------------------------------------------------------
# FOURIER TIME EMBEDDING
#
# Replaces / augments the raw scaled-time input with
#     [sin(2*pi*w_k * t/T_orb), cos(2*pi*w_k * t/T_orb)]  for k = 1..K
#
# NOTE the normalisation is t/T_orb, NOT t/T0 and NOT raw t. That matters:
# T_orb differs per satellite, so integer w_k line up with orbital phase for
# every satellite in the population. A fixed frequency on raw t would target
# one specific altitude only.
#
# Measured on the ablation testbed (position residual, per-satellite fit):
#     h1 only                                 72.2 %
#     h1 + Fourier(t/T_orb) k=1,2             93.4 %
#     h1 + Fourier(t/T_orb) k=1..4            99.1 %
#     h1 + Fourier(t/T_orb) k=1..6            99.1 %   (no gain over k=4)
#     h1 + Fourier at log spacing 1,2,4,8     93.4 %   (worse: skips k=3)
#     Fourier k=1..4 with NO sin u / cos u    98.3 %   (can substitute for h1)
#
# So INTEGER spacing 1..4 is the configuration to use. Log spacing is the
# NeRF default and is measurably the wrong choice here, because it skips the
# third harmonic, which the harmonic ladder showed is load-bearing.
USE_FOURIER_TIME = False
FOURIER_N_FREQS  = 4
FOURIER_SPACING  = 'integer'   # 'integer' | 'log' | 'geometric'
FOURIER_MAX_FREQ = 50.0        # only used by 'geometric' (reference default scale)
FOURIER_LEARNABLE = False      # make the frequencies trainable parameters

# Populated by build_net(). build_network_input() is a free function, so the
# embedding module has to be reachable from module scope; it is ALSO registered
# as a submodule of the network, which is what puts its parameters into
# state_dict() and into the optimizer.
FOURIER_EMBEDDING = None


class FourierEmbedding(nn.Module):
    """Fourier features of orbital phase fraction.

    Adapted from a single-orbit polar reference implementation. Three
    deliberate differences, each for a reason:

    1. INPUT IS t / T_orb, NOT RAW t.
       The reference multiplies raw canonical time by the frequencies. That is
       fine for one orbit, but this model trains on a population: T_orb runs
       5554 s at 400 km to 6827 s at 1400 km, so a fixed frequency on raw t
       means a different fraction of a revolution at every altitude, and the
       network would have to undo that per satellite. Dividing by T_orb first
       makes integer w mean 'w cycles per revolution' for every satellite.

    2. FACTOR OF 2*pi.
       With it, w=1 is exactly one cycle per orbit, which lines the basis up
       with the sin u / cos u harmonics.

    3. RAW t IS NOT REPEATED HERE.
       The reference returns 2K+1 columns because it concatenates raw t.
       t_scaled is already column 0 of the feature vector, immediately before
       this block, so the totals match and repeating it would be a duplicate
       column.

    Two fixes to the reference are carried over as options:

    * Its default scale=1.0 makes exp(linspace(log 1, log 1, K)) return K
      IDENTICAL frequencies, collapsing the embedding to a single sin/cos pair.
      FOURIER_MAX_FREQ defaults to 50.0 and 'geometric' rejects scale <= 1.
    * Its docstring says the frequencies are learnable, but register_buffer
      makes them fixed. FOURIER_LEARNABLE=True does what the comment intended.

    Measured on the ablation testbed (position residual, per-satellite fit):
        integer 1..4   99.1 %      <- best
        log 1,2,4,8    93.4 %      (skips k=3, which the ladder showed matters)
        h1 only        72.2 %
    """

    def __init__(self, n_freqs=4, spacing='integer', max_freq=50.0, learnable=False):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.spacing = spacing
        if self.n_freqs < 1:
            raise ValueError("n_freqs must be at least 1")

        if spacing == 'integer':
            freqs = torch.arange(1, self.n_freqs + 1, dtype=torch.float32)
        elif spacing == 'log':
            freqs = 2.0 ** torch.arange(self.n_freqs, dtype=torch.float32)
        elif spacing == 'geometric':
            if max_freq <= 1.0:
                raise ValueError(
                    f"geometric spacing needs max_freq > 1, got {max_freq}. "
                    "At max_freq=1 every frequency is identical and the "
                    "embedding collapses to a single sin/cos pair."
                )
            freqs = torch.exp(torch.linspace(math.log(1.0), math.log(float(max_freq)),
                                             self.n_freqs))
        else:
            raise ValueError(f"spacing must be integer/log/geometric, got {spacing!r}")

        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        """Columns produced. Raw t is supplied separately by the caller, so
        this is 2K rather than the reference's 2K+1."""
        return 2 * self.n_freqs

    def forward(self, t_over_torb):
        """t_over_torb: [B, 1] elapsed revolutions -> [B, 2K].

        Column order is GROUPED, matching the reference's
        cat([t, sin(z), cos(z)]):

            [sin w1 .. sin wK, cos w1 .. cos wK]

        not interleaved. Checkpoints encode this positionally, so do not
        reorder it.
        """
        z = 2.0 * math.pi * t_over_torb * self.freqs.to(t_over_torb.dtype)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


def _fourier_frequencies():
    """Frequencies the CURRENT flags imply. Used for dimension arithmetic and
    by the verification harness; the live module owns the real tensor."""
    if FOURIER_EMBEDDING is not None:
        return [float(x) for x in FOURIER_EMBEDDING.freqs.detach().cpu()]
    k = int(FOURIER_N_FREQS)
    if k <= 0:
        return []
    if FOURIER_SPACING == 'log':
        return [2.0 ** i for i in range(k)]
    if FOURIER_SPACING == 'geometric':
        return [float(x) for x in
                torch.exp(torch.linspace(math.log(1.0), math.log(float(FOURIER_MAX_FREQ)), k))]
    if FOURIER_SPACING == 'integer':
        return [float(i) for i in range(1, k + 1)]
    raise ValueError(f"FOURIER_SPACING must be integer/log/geometric, got {FOURIER_SPACING!r}")


def make_fourier_embedding():
    return FourierEmbedding(n_freqs=FOURIER_N_FREQS, spacing=FOURIER_SPACING,
                            max_freq=FOURIER_MAX_FREQ, learnable=FOURIER_LEARNABLE)


def _fourier_time_features(t_batch, x0_batch, u0_batch):
    """[B, 2K] Fourier features of elapsed revolutions.

    Differentiable in t, so the physics branch still works: T_orb depends only
    on the initial condition, never on t.
    """
    if not USE_FOURIER_TIME:
        return None
    embed = FOURIER_EMBEDDING if FOURIER_EMBEDDING is not None else make_fourier_embedding()
    T_orb = _orbital_period_torch(x0_batch, u0_batch).clamp_min(1e-6)
    return embed(t_batch / T_orb)


def n_fourier_features():
    if not USE_FOURIER_TIME:
        return 0
    if FOURIER_EMBEDDING is not None:
        return FOURIER_EMBEDDING.out_dim
    return 2 * int(FOURIER_N_FREQS)

def _lat_features(r_batch):
    """Compute [sin(lat), sin^2(lat)] from a position batch.

    lat = arcsin(z / |r|), valid identically in EME2000 and ECEF (both share
    the polar z-axis & equatorial plane). No frame conversion needed because
    J2 is longitude-independent.

    Args:
        r_batch: [B, 3] position in meters (EME2000)
    Returns:
        [B, 2] = [sin_lat, sin2_lat]
    """
    r_norm = torch.norm(r_batch, dim=1, keepdim=True).clamp_min(1.0)
    sin_lat = r_batch[:, 2:3] / r_norm
    sin2_lat = sin_lat * sin_lat
    return torch.cat([sin_lat, sin2_lat], dim=1)

def _kepler_propagate_torch(x0, u0, t, mu=MU_EARTH, newton_iters=10):
    """
    Batched, differentiable Kepler propagator in PyTorch.

    Propagates from Cartesian state (x0, u0) for duration t using
    the universal variable / Kepler equation approach in orbital elements.

    Args:
        x0: [B, 3] initial position (meters)
        u0: [B, 3] initial velocity (m/s)
        t:  [B, 1] propagation time (seconds)
        mu: gravitational parameter (m^3/s^2)
        newton_iters: number of Newton iterations for Kepler's equation

    Returns:
        r_final: [B, 3] propagated position (in the same dtype as the inputs)
    """
    orig_dtype = x0.dtype

    r_vec = x0.to(KEPLER_DTYPE)
    v_vec = u0.to(KEPLER_DTYPE)
    t     = t.to(KEPLER_DTYPE)

    eps = 1e-12

    r_norm = torch.norm(r_vec, dim=1, keepdim=True).clamp_min(eps)
    v_sq = (v_vec * v_vec).sum(dim=1, keepdim=True)

    h_vec = torch.cross(r_vec, v_vec, dim=1)
    h_norm = torch.norm(h_vec, dim=1, keepdim=True).clamp_min(eps)

    p = (h_norm ** 2) / mu

    inv_a = 2.0 / r_norm - v_sq / mu
    a = 1.0 / inv_a.clamp_min(eps)

    vxh = torch.cross(v_vec, h_vec, dim=1)
    e_vec = vxh / mu - r_vec / r_norm
    e = torch.norm(e_vec, dim=1, keepdim=True).clamp_min(eps)

    rdotv = (r_vec * v_vec).sum(dim=1, keepdim=True)

    cos_nu = ((p / r_norm) - 1.0) / e
    cos_nu = cos_nu.clamp(-1.0 + eps, 1.0 - eps)
    nu = torch.acos(cos_nu)

    nu = torch.where(rdotv < 0, 2.0 * math.pi - nu, nu)

    half_E = torch.atan2(
        torch.sqrt((1.0 - e).clamp_min(eps)) * torch.sin(nu / 2.0),
        torch.sqrt((1.0 + e).clamp_min(eps)) * torch.cos(nu / 2.0)
    )
    E0 = 2.0 * half_E

    E0 = E0 % (2.0 * math.pi)

    M0 = E0 - e * torch.sin(E0)

    n = torch.sqrt(mu / (a ** 3).clamp_min(eps))
    M = (M0 + n * t) % (2.0 * math.pi)

    E = M.clone()
    for _ in range(newton_iters):
        f_E = E - e * torch.sin(E) - M
        fp_E = 1.0 - e * torch.cos(E)
        E = E - f_E / fp_E.clamp_min(eps)

    half_nu_new = torch.atan2(
        torch.sqrt((1.0 + e).clamp_min(eps)) * torch.sin(E / 2.0),
        torch.sqrt((1.0 - e).clamp_min(eps)) * torch.cos(E / 2.0)
    )
    nu_new = 2.0 * half_nu_new

    r_new_norm = a * (1.0 - e * torch.cos(E))

    e_hat = e_vec / e
    h_hat = h_vec / h_norm
    q_hat = torch.cross(h_hat, e_hat, dim=1)

    cos_nu_new = torch.cos(nu_new)
    sin_nu_new = torch.sin(nu_new)

    r_final = r_new_norm * (cos_nu_new * e_hat + sin_nu_new * q_hat)

    return r_final.to(orig_dtype)

def _kepler_j2_secular_propagate_torch(x0, u0, t, mu=MU_EARTH, newton_iters=10):
    """
    Kepler propagator with secular J2 corrections (Brouwer first-order).
    Captures node regression (dΩ/dt), apsidal advance (dω/dt),
    and mean motion correction — works correctly for multi-orbit propagation.

    Args:
        x0: [B, 3] initial position (meters)
        u0: [B, 3] initial velocity (m/s)
        t:  [B, 1] propagation time (seconds)

    Returns:
        r_final: [B, 3] propagated position (in the same dtype as the inputs)
    """
    orig_dtype = x0.dtype
    r_vec = x0.to(KEPLER_DTYPE)
    v_vec = u0.to(KEPLER_DTYPE)
    t = t.to(KEPLER_DTYPE)
    eps = 1e-12

    r_norm = torch.norm(r_vec, dim=1, keepdim=True).clamp_min(eps)
    v_sq = (v_vec * v_vec).sum(dim=1, keepdim=True)

    h_vec = torch.cross(r_vec, v_vec, dim=1)
    h_norm = torch.norm(h_vec, dim=1, keepdim=True).clamp_min(eps)

    p = (h_norm ** 2) / mu
    inv_a = 2.0 / r_norm - v_sq / mu
    a = 1.0 / inv_a.clamp_min(eps)

    vxh = torch.cross(v_vec, h_vec, dim=1)
    e_vec = vxh / mu - r_vec / r_norm
    e = torch.norm(e_vec, dim=1, keepdim=True).clamp_min(eps)

    rdotv = (r_vec * v_vec).sum(dim=1, keepdim=True)

    cos_nu = ((p / r_norm) - 1.0) / e
    cos_nu = cos_nu.clamp(-1.0 + eps, 1.0 - eps)
    nu = torch.acos(cos_nu)
    nu = torch.where(rdotv < 0, 2.0 * math.pi - nu, nu)

    half_E = torch.atan2(
        torch.sqrt((1.0 - e).clamp_min(eps)) * torch.sin(nu / 2.0),
        torch.sqrt((1.0 + e).clamp_min(eps)) * torch.cos(nu / 2.0)
    )
    E0 = (2.0 * half_E) % (2.0 * math.pi)
    M0 = E0 - e * torch.sin(E0)

    cos_i = h_vec[:, 2:3] / h_norm
    sin_i_sq = (1.0 - cos_i**2).clamp_min(0.0)

    if USE_OSC2MEAN:
        eta = torch.sqrt((1.0 - e**2).clamp_min(eps))
        ar3 = (a / r_norm) ** 3

        node = torch.cat([-h_vec[:, 1:2], h_vec[:, 0:1],
                          torch.zeros_like(h_vec[:, 0:1])], dim=1)
        node_norm = torch.norm(node, dim=1, keepdim=True)
        n_hat = node / node_norm.clamp_min(eps)
        r_hat = r_vec / r_norm
        h_hat_tmp = h_vec / h_norm
        cos_u = (r_hat * n_hat).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        sin_u = (r_hat * torch.cross(h_hat_tmp, n_hat, dim=1)
                 ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        cos_2u = cos_u * cos_u - sin_u * sin_u

        cos_2u = torch.where(node_norm > 1e-9, cos_2u, torch.ones_like(cos_2u))
        delta_a = (J2 * (R_EARTH**2) / a) * (
            (1.0 - 1.5 * sin_i_sq) * (ar3 - eta**-3)
            + 1.5 * sin_i_sq * ar3 * cos_2u
        )
        a_rate = a - delta_a
        p_rate = a_rate * (1.0 - e**2)
    else:
        a_rate = a
        p_rate = p

    n = torch.sqrt(mu / (a_rate ** 3).clamp_min(eps))
    j2_factor = 1.5 * n * J2 * (R_EARTH**2) / (p_rate**2)

    dOmega_dt = -j2_factor * cos_i
    domega_dt = j2_factor * (2.0 - 2.5 * sin_i_sq)

    e_sq = e**2
    n_corrected = n + j2_factor * torch.sqrt((1.0 - e_sq).clamp_min(eps)) * (1.0 - 1.5 * sin_i_sq)

    M = (M0 + n_corrected * t) % (2.0 * math.pi)

    delta_Omega = dOmega_dt * t
    delta_omega = domega_dt * t

    E = M.clone()
    for _ in range(newton_iters):
        f_E = E - e * torch.sin(E) - M
        fp_E = 1.0 - e * torch.cos(E)
        E = E - f_E / fp_E.clamp_min(eps)

    half_nu_new = torch.atan2(
        torch.sqrt((1.0 + e).clamp_min(eps)) * torch.sin(E / 2.0),
        torch.sqrt((1.0 - e).clamp_min(eps)) * torch.cos(E / 2.0)
    )
    nu_new = 2.0 * half_nu_new

    r_new_norm = a * (1.0 - e * torch.cos(E))

    e_hat = e_vec / e
    h_hat = h_vec / h_norm
    q_hat = torch.cross(h_hat, e_hat, dim=1)

    cos_dw = torch.cos(delta_omega)
    sin_dw = torch.sin(delta_omega)
    e_hat_rot = cos_dw * e_hat + sin_dw * q_hat
    q_hat_rot = -sin_dw * e_hat + cos_dw * q_hat

    cos_dO = torch.cos(delta_Omega)
    sin_dO = torch.sin(delta_Omega)

    e_hat_final = torch.cat([
        cos_dO * e_hat_rot[:, 0:1] - sin_dO * e_hat_rot[:, 1:2],
        sin_dO * e_hat_rot[:, 0:1] + cos_dO * e_hat_rot[:, 1:2],
        e_hat_rot[:, 2:3]
    ], dim=1)

    q_hat_final = torch.cat([
        cos_dO * q_hat_rot[:, 0:1] - sin_dO * q_hat_rot[:, 1:2],
        sin_dO * q_hat_rot[:, 0:1] + cos_dO * q_hat_rot[:, 1:2],
        q_hat_rot[:, 2:3]
    ], dim=1)

    cos_nu_new = torch.cos(nu_new)
    sin_nu_new = torch.sin(nu_new)
    r_final = r_new_norm * (cos_nu_new * e_hat_final + sin_nu_new * q_hat_final)

    return r_final.to(orig_dtype)

def earth_gravity_acc_torch(r):
    """Central + J2 gravity in EME2000. r: [B,3] or [3], meters."""
    if r.dim() == 1:
        r = r.unsqueeze(0)

    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]

    eps_m = torch.tensor(1e-3, dtype=r.dtype, device=r.device)
    r2 = (r * r).sum(dim=-1, keepdim=True).clamp_min(eps_m**2)
    r_norm  = torch.sqrt(r2)
    inv_r2 = 1.0 / r2
    inv_r3 = inv_r2 / r_norm
    inv_r5 = inv_r3 * inv_r2

    a_central = -MU_EARTH * r * inv_r3

    a_J2=0
    if USE_J2:
        z2 = z * z
        factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) * inv_r5
        common_xy = (5.0 * z2 * inv_r2 - 1.0)
        ax = factor * x * common_xy
        ay = factor * y * common_xy
        az = factor * z * (5.0 * z2 * inv_r2 - 3.0)
        a_J2 = torch.cat([ax, ay, az], dim=1)

    return a_central + a_J2

def solar_gravity_acc_torch(r, sun_pos): return 0
def lunar_gravity_acc_torch(r, moon_pos): return 0
def drag_acc_torch(v, rho, C_d, A, m): return 0
def solar_radiation_acc_torch(theta, A, epsilon, sun_dir, normal): return 0

class Net(nn.Module):
    """
    MLP: [t̂, x̂0(3), û0(3)] -> r̂_nn(t̂) in R^3
    (Network predicts scaled position directly)
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for layer in self.linear[:-1]:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.linear[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.linear[-1].bias)

    def forward(self, x):
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers)-2):
            a = self.activation(self.linear[i](a))
        return self.linear[-1](a)

# ---------------------------------------------------------------------------
# QUANTUM LAYER (QAPINN)
#
# OFF by default. Import of PennyLane is deferred so this module still loads
# on machines without it; the error only fires if a QNet is actually built.
#
# WHY THIS DIFFERS FROM THE ORIGINAL QNet
# ---------------------------------------
# The original required layers[0] == 9 exactly, because it angle-embedded the
# 9 raw inputs onto 9 wires and read out qml.probs on 8 wires -> 2^8 = 256.
# That constraint breaks as soon as any feature flag is on: with harmonics=4
# the input is 20 wide, which would need 20 qubits and produce 2^19 = 524288
# readout values. Not simulable.
#
# The fix is a classical projection layer in front of the circuit, mapping
# input_dim -> n_qubits. The quantum register stays at a fixed, simulable
# width regardless of how many classical features are enabled. When
# layers[0] == n_qubits the projection is skipped entirely, so the original
# 9-input behaviour and its state-dict layout are reproduced exactly.
#
# GRADIENTS: qml.probs is incompatible with adjoint differentiation. For
# training use diff_method='parameter-shift'. inference_only=True selects
# diff_method=None, which is much faster but cannot be backpropagated through.
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    """Hybrid quantum-classical residual network, two placements.

    VARIANT A -- quantum as the INPUT layer   (USE_QUANTUM_INPUT)
        features -> [projection] -> circuit -> classical -> ... -> 3
        The circuit sees the physical features directly, as in the reference.

    VARIANT B -- quantum as a HIDDEN layer    (USE_QUANTUM_HIDDEN)
        features -> classical -> tanh -> ... -> [projection] -> circuit
                 -> classical -> ... -> 3
        Classical layers extract a representation first; the circuit operates
        on that. Set QUANTUM_HIDDEN_POSITION to choose which layer it replaces.

    PROJECTION, AND WHY IT IS NEEDED
    --------------------------------
    AngleEmbedding puts one feature on one wire, so the reference requires
    layers[0] == n_qubits exactly. That is fine at 9 raw inputs but impossible
    once features are enabled: harmonics=4 gives 20 inputs, needing 20 qubits
    and a 2**19 = 524288 readout, which is not simulable. A small linear
    projection maps the incoming width down to n_qubits, keeping the register
    at a fixed simulable size no matter how many classical features are on.
    When the widths already match, the projection is skipped and the behaviour
    is byte-for-byte the reference.

    The projection output is squashed into [-pi, pi] with pi*tanh, because
    AngleEmbedding angles that wander outside that range wrap around and make
    the encoding non-injective.

    DIFFERENTIATION
    ---------------
    default.qubit with diff_method='backprop', per the reference. Unlike
    parameter-shift this supports BROADCASTED tapes, so a whole batch goes
    through in one call and no per-sample loop is needed.

    CHECKPOINTS
    -----------
    The classical stack is stored under self.linear, matching Net, so a
    classical checkpoint partially loads with strict=False and only the
    quantum layer starts fresh.
    """

    def __init__(self, layers, n_qubits=9, q_depth=4, quantum_position=0,
                 quantum_batch_size=None):
        super().__init__()
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "The quantum path requires PennyLane. Install it, or leave "
                "USE_QUANTUM_INPUT and USE_QUANTUM_HIDDEN False for the "
                "classical network."
            ) from exc

        self.layers = list(layers)
        self.activation = nn.Tanh()
        self.n_qubits = int(n_qubits)
        self.q_depth = int(q_depth)
        self.quantum_batch_size = quantum_batch_size

        n_weight_layers = len(self.layers) - 1
        if not 0 <= int(quantum_position) < n_weight_layers - 1:
            raise ValueError(
                f"quantum_position must be in [0, {n_weight_layers - 2}] for "
                f"layers={self.layers} (it cannot be the output layer), got "
                f"{quantum_position}"
            )
        self.quantum_position = int(quantum_position)

        readout_wires = self.n_qubits - 1
        expected = 2 ** readout_wires
        out_width = self.layers[self.quantum_position + 1]
        if out_width != expected:
            raise ValueError(
                f"layers[{self.quantum_position + 1}] must be 2**(n_qubits-1) "
                f"= {expected} for n_qubits={self.n_qubits}, got {out_width}. "
                f"Either set that width to {expected}, or use n_qubits="
                f"{int(math.log2(out_width)) + 1}."
            )

        in_width = self.layers[self.quantum_position]
        if in_width != self.n_qubits:
            self.input_projection = nn.Linear(in_width, self.n_qubits)
            nn.init.xavier_normal_(self.input_projection.weight, gain=1.0)
            nn.init.zeros_(self.input_projection.bias)
        else:
            self.input_projection = None

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(readout_wires))

        # Same attribute name as Net so classical checkpoints partially load.
        self.linear = nn.ModuleList()
        for i in range(n_weight_layers):
            if i == self.quantum_position:
                self.linear.append(
                    qml.qnn.TorchLayer(circuit, {"weights": (self.q_depth, self.n_qubits)}))
            else:
                self.linear.append(nn.Linear(self.layers[i], self.layers[i + 1]))

        for i, layer in enumerate(self.linear[:-1]):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        if isinstance(self.linear[-1], nn.Linear):
            nn.init.normal_(self.linear[-1].weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.linear[-1].bias)

    @property
    def variant(self):
        return "A_quantum_input" if self.quantum_position == 0 else "B_quantum_hidden"

    @property
    def recommended_torch_device(self):
        """default.qubit with backprop is a PyTorch passthrough, so unlike the
        Lightning backends it follows the tensors and CUDA is fine."""
        return DEVICE

    def _quantum(self, x):
        layer = self.linear[self.quantum_position]
        if self.input_projection is not None:
            x = math.pi * torch.tanh(self.input_projection(x))
        cs = self.quantum_batch_size
        if x.ndim == 1 or not cs or cs <= 0 or x.shape[0] <= cs:
            return layer(x)
        # backprop broadcasts fine; chunking only bounds simulator memory.
        return torch.cat([layer(c) for c in x.split(cs, dim=0)], dim=0)

    def forward(self, x):
        a = x
        for i in range(len(self.linear) - 1):
            a = self._quantum(a) if i == self.quantum_position else self.linear[i](a)
            a = self.activation(a)
        return self.linear[-1](a)


# ---------------------------------------------------------------------------
# NETWORK SELECTION
#
# Both quantum variants are OFF by default. They are mutually exclusive: the
# circuit sits at exactly one depth in the stack.
# ---------------------------------------------------------------------------
USE_QUANTUM_INPUT  = False    # Variant A: circuit is the input layer
USE_QUANTUM_HIDDEN = False    # Variant B: circuit is a hidden layer
QUANTUM_HIDDEN_POSITION = 1   # which layer index the circuit replaces in B
QUANTUM_N_QUBITS = 9
QUANTUM_DEPTH = 4             # BasicEntanglerLayers repetitions
QUANTUM_BATCH_SIZE = None     # None = one broadcast call; set an int to bound memory


def quantum_readout_width(n_qubits=None):
    """Width the circuit emits: probs over n_qubits-1 wires."""
    return 2 ** ((QUANTUM_N_QUBITS if n_qubits is None else int(n_qubits)) - 1)


def build_net(layers):
    """Construct the network implied by the current flags, already placed.

    Raises if both quantum variants are on, since the circuit occupies exactly
    one position in the stack.
    """
    if USE_QUANTUM_INPUT and USE_QUANTUM_HIDDEN:
        raise ValueError(
            "USE_QUANTUM_INPUT and USE_QUANTUM_HIDDEN are mutually exclusive: "
            "the circuit occupies one layer. Pick a variant."
        )
    if USE_QUANTUM_INPUT or USE_QUANTUM_HIDDEN:
        position = 0 if USE_QUANTUM_INPUT else int(QUANTUM_HIDDEN_POSITION)
        net = QNet(layers, n_qubits=QUANTUM_N_QUBITS, q_depth=QUANTUM_DEPTH,
                   quantum_position=position, quantum_batch_size=QUANTUM_BATCH_SIZE)
        return net.to(device=net.recommended_torch_device, dtype=DTYPE)
    return Net(layers).to(device=DEVICE, dtype=DTYPE)


def attach_fourier(net):
    """Create the Fourier module, register it on the net, and publish it.

    Registering it as a submodule is what puts learnable frequencies into
    state_dict() and into the optimizer; the module-level handle is what the
    free function build_network_input() reads.
    """
    global FOURIER_EMBEDDING
    if not USE_FOURIER_TIME:
        FOURIER_EMBEDDING = None
        return net
    embed = make_fourier_embedding().to(device=next(net.parameters()).device, dtype=DTYPE)
    net.fourier_embedding = embed
    FOURIER_EMBEDDING = embed
    return net


def _orbital_period_torch(x0, u0, mu=MU_EARTH):
    """Keplerian orbital period from the IC. Returns [B,1] seconds.

    1/a = 2/r - v^2/mu (vis-viva), then T = 2*pi*sqrt(a^3/mu).
    Per-sample by construction, so any gate built on it is per-satellite.
    """
    eps = 1e-12
    rn = torch.norm(x0, dim=1, keepdim=True).clamp_min(eps)
    v2 = (u0 * u0).sum(dim=1, keepdim=True)
    inv_a = (2.0 / rn - v2 / mu).clamp_min(eps)
    a = 1.0 / inv_a
    return 2.0 * math.pi * torch.sqrt((a ** 3 / mu).clamp_min(eps))

def _compute_gate(t_scaled, x0_batch, u0_batch, T0, Tseg_batch=None):
    """Hard-constraint gate in [0,1]; single entry point for every variant.

    Built as a function of t_scaled (NOT raw t) so autograd yields the correct
    d/d(t_scaled) used by the velocity path in _time_derivatives.

    All variants satisfy gate(0)=0 (r(0)=r0 exact) and dgate/dt(0)=0
    (baseline velocity at t=0 preserved).
    """
    if GATE_KIND == 'legacy':
        if Tseg_batch is not None:
            T_max_scaled = (Tseg_batch / T0).clamp_min(1e-6)
        else:
            T_max_scaled = torch.full_like(t_scaled, float(T_SCALED_MAX)).clamp_min(1e-6)
        return (t_scaled / T_max_scaled) ** GATE_POWER

    T_orb = _orbital_period_torch(x0_batch, u0_batch)
    if GATE_KIND == 'tanh':
        tau_scaled = ((T_orb / GATE_TAU_DIV) / T0).clamp_min(1e-6)
        return torch.tanh(t_scaled / tau_scaled) ** 2
    if GATE_KIND == 'orbit':
        Torb_scaled = (T_orb / T0).clamp_min(1e-6)
        return (t_scaled / Torb_scaled).clamp(0.0, 1.0) ** GATE_POWER
    raise ValueError(f"Unknown GATE_KIND={GATE_KIND!r}")

def _orbit_features(x0, u0, r_baseline_scaled):
    """Static orbit descriptors plus optional argument-of-latitude harmonics.

    From the IC (constant over the window): cos i, sin^2 i, eccentricity.
    From the BASELINE position at time t: sin u, cos u (orbital phase), which
    is what lets the network represent the once-per-orbit along-track
    oscillation. Scale-invariant in r_baseline_scaled (only ratios are used),
    so passing the R0-scaled baseline position is correct.

    Guards: near-equatorial orbits have an undefined node direction; sin_u and
    cos_u are then replaced by constants. Harmless, because the cross-track
    signal they inform vanishes as sin i -> 0.
    """
    eps = 1e-9
    rn0 = torch.norm(x0, dim=1, keepdim=True).clamp_min(eps)

    h = torch.cross(x0, u0, dim=1)
    hn = torch.norm(h, dim=1, keepdim=True).clamp_min(eps)
    cos_i = (h[:, 2:3] / hn).clamp(-1.0, 1.0)
    sin2_i = (1.0 - cos_i * cos_i).clamp_min(0.0)
    sin_i = torch.sqrt(sin2_i.clamp_min(1e-12))

    e_vec = torch.cross(u0, h, dim=1) / MU_EARTH - x0 / rn0
    ecc = torch.norm(e_vec, dim=1, keepdim=True)

    if not USE_ORBIT_PHASE_FEATURES:
        return torch.cat([cos_i, sin2_i, ecc], dim=1)

    node = torch.cat([-h[:, 1:2], h[:, 0:1], torch.zeros_like(h[:, 0:1])], dim=1)
    node_n = torch.norm(node, dim=1, keepdim=True)
    n_hat = node / node_n.clamp_min(eps)

    rb_n = torch.norm(r_baseline_scaled, dim=1, keepdim=True).clamp_min(eps)
    sin_phi = r_baseline_scaled[:, 2:3] / rb_n
    sin_u = (sin_phi / sin_i).clamp(-1.0, 1.0)
    cos_u = ((r_baseline_scaled * n_hat).sum(dim=1, keepdim=True) / rb_n).clamp(-1.0, 1.0)

    ok = (node_n > 1e-6) & (sin_i > 1e-6)
    sin_u = torch.where(ok, sin_u, torch.zeros_like(sin_u))
    cos_u = torch.where(ok, cos_u, torch.ones_like(cos_u))

    out = [cos_i, sin2_i, ecc, sin_u, cos_u]

    # Higher harmonics of orbital phase by the angle-addition recursion
    #   sin(ku) = sin((k-1)u) cos u + cos((k-1)u) sin u
    #   cos(ku) = cos((k-1)u) cos u - sin((k-1)u) sin u
    # No new trigonometry, no division, and no new guards: every harmonic
    # inherits the equatorial guard through sin_u / cos_u above.
    #
    # Why these and not a Fourier embedding of time: the residual is periodic in
    # ORBITAL PHASE, not in t. Over 80000 s (about 13 revolutions) u advances at
    # a rate that differs from 2*pi*t/T_orb through apsidal advance and
    # eccentricity, so a time-based embedding accumulates phase drift that a
    # u-based one does not.
    sk, ck = sin_u, cos_u
    for _ in range(2, int(ORBIT_N_HARMONICS) + 1):
        sk, ck = sk * cos_u + ck * sin_u, ck * cos_u - sk * sin_u
        out.extend([sk, ck])

    return torch.cat(out, dim=1)

def build_network_input(t_batch, x0_batch, u0_batch, r_baseline_scaled,
                       t_scaled, R0, V0):
    """THE single place where the network input vector is assembled.

    Every call site (training forward, predict_step, predict_step_batch) routes
    through here. Previously the feature list was duplicated at five sites,
    which meant any new feature had to be added five times and a missed site
    failed silently with a shape error at load time. Do not reintroduce that:
    if a feature is needed, add it here only.

    Column order is fixed and must never be reordered, because checkpoints
    encode it positionally:

        [ time block | x0/R0 (3) | u0/V0 (3) | R0/Re | V0/V0_SCALE
          | latitude (2, optional) | orbit block (3 or 3 + 2N, optional) ]

    The time block is t_scaled alone, or t_scaled followed by the 2K Fourier
    columns when USE_FOURIER_TIME is set.
    """
    R0_scaled_in = R0 / R_EARTH
    V0_scaled_in = V0 / V0_SCALE

    feat = [t_scaled]
    if USE_FOURIER_TIME:
        fe = _fourier_time_features(t_batch, x0_batch, u0_batch)
        if fe is not None:
            feat.append(fe)
    feat.extend([x0_batch / R0, u0_batch / V0, R0_scaled_in, V0_scaled_in])

    if USE_LAT_FEATURES:
        feat.append(_lat_features(r_baseline_scaled))
    if USE_ORBIT_FEATURES:
        feat.append(_orbit_features(x0_batch, u0_batch, r_baseline_scaled))

    return torch.cat(feat, dim=1)


def input_dim():
    """Network input width implied by the CURRENT module flags.

    train_opti.py, the evaluation scripts and any checkpoint loader must all
    call this rather than recomputing the arithmetic, otherwise they can drift
    apart and the mismatch only surfaces as a load_state_dict shape error.
    """
    dim = 9                                   # t, x0(3), u0(3), R0, V0
    dim += n_fourier_features()
    if USE_LAT_FEATURES:
        dim += _N_LAT_FEATURES
    if USE_ORBIT_FEATURES:
        dim += 3
        if USE_ORBIT_PHASE_FEATURES:
            dim += 2 * int(ORBIT_N_HARMONICS)
    return dim


def describe_config():
    """One-line summary of every flag that changes the network or the loss."""
    return (f"REGIME={REGIME} GATE_KIND={GATE_KIND} TAU_DIV={GATE_TAU_DIV} "
            f"GATE_POWER={GATE_POWER} J2_BASE={USE_J2_BASELINE} "
            f"OSC2MEAN={USE_OSC2MEAN} LAT={USE_LAT_FEATURES} "
            f"ORBIT={USE_ORBIT_FEATURES} ORBIT_PHASE={USE_ORBIT_PHASE_FEATURES} "
            f"HARMONICS={ORBIT_N_HARMONICS} "
            f"FOURIER={USE_FOURIER_TIME}({FOURIER_N_FREQS},{FOURIER_SPACING},"
            f"learn={FOURIER_LEARNABLE}) "
            f"QINPUT={USE_QUANTUM_INPUT} QHIDDEN={USE_QUANTUM_HIDDEN} "
            f"USE_J2_PHYS={USE_J2} PDE_NORM={PDE_NORM} "
            f"PDE_GATED={PDE_GATE_WEIGHTED} PDE_SUB_KEPLER={PDE_SUBTRACT_KEPLER} "
            f"input_dim={input_dim()}")


class GeneralistModel:
    """
    KEPLER BASELINE HARD CONSTRAINT MODEL
    - Hard constraint: r(t) = kepler(x0, u0, t) + t^3 * N(t)
    - Network learns the residual correction on top of Kepler propagation.
    - Loss is a weighted sum: loss = (W_PDE * loss_pde) + (W_DATA * loss_data)
    """
    def __init__(self, net: nn.Module, W_DATA: float, W_PDE: float):
        self.net = net
        self.W_DATA = W_DATA
        self.W_PDE = W_PDE

    def _time_derivative(self, y_scaled, t_scaled, create_graph=True):
        """Vector time derivative d(y_scaled)/d(t_scaled).

        Args:
            y_scaled: [B, 3]
            t_scaled: [B, 1]
            create_graph: must stay True during training if the resulting
                derivative participates in the loss, otherwise gradients will
                not flow back to the network through that derivative term.
        """
        comps = [
            torch.autograd.grad(
                y_scaled[:, i].sum(),
                t_scaled,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            for i in range(y_scaled.shape[1])
        ]

        if any(c is None for c in comps):
            return torch.zeros_like(y_scaled)
        return torch.cat(comps, dim=1)

    def _time_derivatives(self, r_scaled, t_scaled, need_acceleration=True, create_graph=True):
        """Compute dr/dt and optionally d²r/dt².

        Notes:
            - During training, create_graph must be True whenever the returned
              derivative is used inside a loss term.
            - During inference-only velocity evaluation, create_graph can be
              False to save memory.
        """
        drdt_scaled = self._time_derivative(r_scaled, t_scaled, create_graph=create_graph)

        if not need_acceleration:
            return drdt_scaled, None

        d2rdt2_scaled = self._time_derivative(drdt_scaled, t_scaled, create_graph=create_graph)
        return drdt_scaled, d2rdt2_scaled

    def physics_accel(self, r, v, sun_pos, moon_pos, rho, C_d, A, m, theta, epsilon, sun_dir, normal):
        a = earth_gravity_acc_torch(r)
        if USE_SUN_AND_MOON:
            a = a + solar_gravity_acc_torch(r, sun_pos) + lunar_gravity_acc_torch(r, moon_pos)
        if USE_DRAG: a = a + drag_acc_torch(v, rho, C_d, A, m)
        if USE_SRP:  a = a + solar_radiation_acc_torch(theta, A, epsilon, sun_dir, normal)
        return a

    def _kepler_baseline_scaled(self, x0_batch, u0_batch, t_batch, R0):
        """
        Compute baseline propagation in scaled coordinates.

        If USE_J2_BASELINE: uses Kepler + secular J2 (node regression, apsidal
        advance, mean motion correction). Works for multi-orbit propagation.
        Otherwise: pure Kepler only.

        Args:
            x0_batch: [B, 3] initial positions (real units, meters)
            u0_batch: [B, 3] initial velocities (real units, m/s)
            t_batch:  [B, 1] time (real units, seconds)
            R0:       [B, 1] position scale factor

        Returns:
            r_baseline_scaled: [B, 3] baseline position in scaled coordinates
        """
        if USE_J2_BASELINE:
            r_baseline = _kepler_j2_secular_propagate_torch(x0_batch, u0_batch, t_batch)
        else:
            r_baseline = _kepler_propagate_torch(x0_batch, u0_batch, t_batch)
        return r_baseline / R0

    def _forward_scaled_position(self, t_batch, x0_batch, u0_batch, Tseg_batch=None):
        """Shared forward pass up to final scaled position.

        Tseg_batch is per-sample window duration [B, 1] (real seconds).
        When provided, the cubic gate uses (t / Tseg)^3 per sample, instead of
        the global (t_scaled / T_SCALED_MAX)^3. When None, falls back to the
        old global behaviour for backward compatibility.

        This is the common work used by both:
          - data branch: only r and v are needed
          - physics branch: r, v, and a are needed
        """
        if not t_batch.requires_grad:
            t_batch.requires_grad_(True)

        R0 = torch.norm(x0_batch, dim=1, keepdim=True).clamp_min(1.0)
        V0 = torch.norm(u0_batch, dim=1, keepdim=True).clamp_min(1.0)
        T0 = R0 / V0

        t_scaled  = t_batch / T0
        x0_scaled = x0_batch / R0
        u0_scaled = u0_batch / V0

        R0_scaled_in = R0 / R_EARTH
        V0_scaled_in = V0 / V0_SCALE

        r_kepler_scaled = self._kepler_baseline_scaled(x0_batch, u0_batch, t_scaled * T0, R0)

        inp = build_network_input(t_scaled * T0, x0_batch, u0_batch,
                                  r_kepler_scaled, t_scaled, R0, V0)
        r_residual_scaled = self.net(inp)

        gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0, Tseg_batch)
        r_final_scaled = r_kepler_scaled + (gate_pow * r_residual_scaled)

        return r_final_scaled, t_scaled, R0, V0, T0

    def get_scaled_state_rv(self, t_batch, x0_batch, u0_batch, Tseg_batch=None):
        """Efficient path for the supervised data branch.

        Computes:
            r(t), v(t)
        but skips:
            a(t) = d²r/dt²

        This avoids the expensive second derivative whenever physics loss is
        not being evaluated for that batch.
        """
        r_final_scaled, t_scaled, R0, V0, T0 = self._forward_scaled_position(
            t_batch, x0_batch, u0_batch, Tseg_batch=Tseg_batch
        )

        v_final_scaled, _ = self._time_derivatives(
            r_final_scaled, t_scaled,
            need_acceleration=False,
            create_graph=True,
        )

        r_nn = r_final_scaled * R0
        v_nn = v_final_scaled * (R0 / T0)
        return r_nn, v_nn, R0, V0

    def get_scaled_state(self, t_batch, x0_batch, u0_batch, Tseg_batch=None):
        """Full state path for the physics branch.

        Computes:
            r(t), v(t), a(t)
        where:
            v(t) = dr/dt
            a(t) = dv/dt = d²r/dt²
        """
        r_final_scaled, t_scaled, R0, V0, T0 = self._forward_scaled_position(
            t_batch, x0_batch, u0_batch, Tseg_batch=Tseg_batch
        )

        v_final_scaled, a_final_scaled = self._time_derivatives(
            r_final_scaled, t_scaled,
            need_acceleration=True,
            create_graph=True,
        )

        r_nn = r_final_scaled * R0
        v_nn = v_final_scaled * (R0 / T0)
        a_nn = a_final_scaled * (R0 / (T0**2))

        return r_nn, v_nn, a_nn, R0, V0

    def epoch_loss(self,
                   t_data_batch, x0_data_batch, u0_data_batch, r_data_truth_batch, v_data_truth_batch,
                   t_pde_batch, x0_pde_batch, u0_pde_batch,
                   sun_pos, moon_pos, rho, C_d, A, m, theta, epsilon, sun_dir, normal,
                   Tseg_data_batch=None, Tseg_pde_batch=None):

        loss_data = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        loss_pde = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

        if t_data_batch.shape[0] > 0:
            r_data_nn, v_data_nn, R0_data, V0_data = self.get_scaled_state_rv(
                t_data_batch, x0_data_batch, u0_data_batch,
                Tseg_batch=Tseg_data_batch,
            )

            err_data_pos_scaled = (r_data_nn - r_data_truth_batch) / R0_data
            err_data_vel_scaled = (v_data_nn - v_data_truth_batch) / V0_data

            loss_data = torch.mean(err_data_pos_scaled**2) + torch.mean(err_data_vel_scaled**2)

        if t_pde_batch.shape[0] > 0:
            r_pde_nn, v_pde_nn, a_pde_nn, _, _ = self.get_scaled_state(
                t_pde_batch, x0_pde_batch, u0_pde_batch,
                Tseg_batch=Tseg_pde_batch,
            )

            a_true_pde = self.physics_accel(r_pde_nn, v_pde_nn, sun_pos, moon_pos, rho, C_d, A, m, theta, epsilon, sun_dir, normal)

            if PDE_NORM == 'j2':
                if not USE_J2:
                    raise ValueError(
                        "PDE_NORM='j2' requires USE_J2=True. With USE_J2=False the "
                        "J2 reference acceleration is identically zero, the clamp "
                        "fires, and loss_pde diverges to ~1e20."
                    )
                rn_pde = torch.norm(r_pde_nn, dim=1, keepdim=True).clamp_min(1.0)
                a_central_pde = -MU_EARTH * r_pde_nn / rn_pde**3
                scale = torch.norm(a_true_pde - a_central_pde, dim=1, keepdim=True).clamp_min(1e-12)
            else:
                scale = (torch.norm(a_true_pde, dim=1, keepdim=True) + torch.norm(a_pde_nn, dim=1, keepdim=True) + 1e-9)

            if PDE_SUBTRACT_KEPLER:
                # Compare the network's acceleration against the perturbation
                # ALONE, with the two-body part of the baseline removed from
                # both sides. r_base is recomputed here at the PDE collocation
                # times so the subtraction uses the baseline, not r_pred.
                R0_pde_k = torch.norm(x0_pde_batch, dim=1, keepdim=True).clamp_min(1.0)
                with torch.no_grad():
                    r_base_pde = self._kepler_baseline_scaled(
                        x0_pde_batch, u0_pde_batch, t_pde_batch, R0_pde_k) * R0_pde_k
                    rn_bk = torch.norm(r_base_pde, dim=1, keepdim=True).clamp_min(1.0)
                    a_kep_base = -MU_EARTH * r_base_pde / rn_bk**3
                residual_pde = ((a_pde_nn - a_kep_base) - (a_true_pde - a_kep_base)) / scale
            else:
                residual_pde = (a_pde_nn - a_true_pde) / scale

            if PDE_GATE_WEIGHTED:
                R0_pde = torch.norm(x0_pde_batch, dim=1, keepdim=True).clamp_min(1.0)
                V0_pde = torch.norm(u0_pde_batch, dim=1, keepdim=True).clamp_min(1.0)
                T0_pde = R0_pde / V0_pde
                with torch.no_grad():
                    w_pde_pt = _compute_gate(
                        t_pde_batch / T0_pde, x0_pde_batch, u0_pde_batch,
                        T0_pde, Tseg_pde_batch,
                    )
                loss_pde = (w_pde_pt * residual_pde**2).sum() / w_pde_pt.sum().clamp_min(1e-12)
            else:
                loss_pde = torch.mean(residual_pde**2)

        total_loss = (self.W_PDE * loss_pde) + (self.W_DATA * loss_data)

        return total_loss, loss_pde, loss_data

def predict_step(model, x0_np, u0_np, T_total, npts=100, Tseg_train=None):
    """
    Predicts a SINGLE short-arc (r(t), v(t)) on t in [0, T_total] for one IC.
    Uses the Kepler baseline hard constraint model.

    Tseg_train (optional): training-time window length. When set,
    the gate uses (t / Tseg_train)^3 instead of (t / T_total)^3 so that
    train-time and predict-time gates agree. Pass the same WINDOW_DURATION_SEC
    you used in training. When None, falls back to T_total (current behavior).
    """
    model.net.eval()

    x0 = torch.tensor(x0_np, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    u0 = torch.tensor(u0_np, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    t_grid = np.linspace(0.0, T_total, npts).reshape(-1,1)
    t_torch = torch.tensor(t_grid, dtype=DTYPE, device=DEVICE, requires_grad=True)

    N = t_torch.size(0)
    x0_batch = x0.repeat(N, 1)
    u0_batch = u0.repeat(N, 1)

    R0 = torch.norm(x0_batch, dim=1, keepdim=True).clamp_min(1.0)
    V0 = torch.norm(u0_batch, dim=1, keepdim=True).clamp_min(1.0)
    T0 = R0 / V0

    t_scaled  = t_torch / T0
    x0_scaled = x0_batch / R0
    u0_scaled = u0_batch / V0

    R0_scaled_in = R0 / R_EARTH
    V0_scaled_in = V0 / V0_SCALE

    _propagate = _kepler_j2_secular_propagate_torch if USE_J2_BASELINE else _kepler_propagate_torch

    T_gate_ref = float(Tseg_train) if Tseg_train is not None else float(T_total)

    T_max_scaled = (torch.full_like(t_scaled, T_gate_ref) / T0).clamp_min(1e-6)

    with torch.no_grad():
        r_kepler = _propagate(x0_batch, u0_batch, t_torch)
        r_kepler_scaled = r_kepler / R0

        inp = build_network_input(t_torch, x0_batch, u0_batch,
                                  r_kepler_scaled, t_scaled, R0, V0)
        r_nn_scaled = model.net(inp)

        gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0,
                                 torch.full_like(t_scaled, T_gate_ref))
        r_final_scaled = r_kepler_scaled + (gate_pow * r_nn_scaled)
        r = r_final_scaled * R0

    with torch.enable_grad():
        t_scaled_grad = t_torch / T0

        r_kepler_grad = _propagate(x0_batch, u0_batch, t_scaled_grad * T0)
        r_kepler_scaled_grad = r_kepler_grad / R0

        inp_grad = build_network_input(t_scaled_grad * T0, x0_batch, u0_batch,
                                       r_kepler_scaled_grad, t_scaled_grad, R0, V0)
        r_nn_scaled_grad = model.net(inp_grad)

        gate_pow_grad = _compute_gate(t_scaled_grad, x0_batch, u0_batch, T0,
                                      torch.full_like(t_scaled_grad, T_gate_ref))
        r_final_scaled_grad = r_kepler_scaled_grad + (gate_pow_grad * r_nn_scaled_grad)

        v_hat, _ = model._time_derivatives(r_final_scaled_grad, t_scaled_grad, need_acceleration=False, create_graph=False)

    v = v_hat * (R0 / T0)

    t_out = t_grid.astype(np.float64)
    r_out = r.detach().cpu().numpy().astype(np.float64)
    v_out = v.detach().cpu().numpy().astype(np.float64)

    return t_out, r_out, v_out

def predict_step_batch(model, x0_all_np, u0_all_np, T_total, npts=100, compute_velocity=False,
                       Tseg_train=None):
    """
    Batched prediction for S satellites at once.
    Uses the Kepler baseline hard constraint model.

    Args:
        model:            GeneralistModel with loaded weights
        x0_all_np:        [S, 3] initial positions in meters (numpy float64)
        u0_all_np:        [S, 3] initial velocities in meters/s (numpy float64)
        T_total:          propagation duration in seconds
        npts:             number of output time points
        compute_velocity: if True, compute velocity via autograd (slower but exact)
        Tseg_train:       If set, use this as the gate's reference window
                          length (must match the training window length, e.g. 80000.0).
                          Ensures train-time and predict-time gates agree. When None
                          (default), falls back to T_total (each call is treated as
                          its own self-contained window).

    Returns:
        t_out:  [npts, 1]     time grid in seconds (shared by all sats)
        r_out:  [S, npts, 3]  predicted positions in meters
        v_out:  [S, npts, 3]  predicted velocities in m/s (only if compute_velocity=True, else None)
    """
    model.net.eval()

    S = x0_all_np.shape[0]

    T_gate_ref = float(Tseg_train) if Tseg_train is not None else float(T_total)

    t_grid = np.linspace(0.0, T_total, npts).reshape(-1, 1)

    x0_all = torch.tensor(x0_all_np, dtype=DTYPE, device=DEVICE)
    u0_all = torch.tensor(u0_all_np, dtype=DTYPE, device=DEVICE)
    t_torch = torch.tensor(t_grid, dtype=DTYPE, device=DEVICE,
                           requires_grad=compute_velocity)

    x0_batch = x0_all.repeat_interleave(npts, dim=0)
    u0_batch = u0_all.repeat_interleave(npts, dim=0)

    t_batch = t_torch.repeat(S, 1)

    R0 = torch.norm(x0_batch, dim=1, keepdim=True).clamp_min(1.0)
    V0 = torch.norm(u0_batch, dim=1, keepdim=True).clamp_min(1.0)
    T0 = R0 / V0

    t_scaled  = t_batch / T0
    x0_scaled = x0_batch / R0
    u0_scaled = u0_batch / V0
    R0_scaled_in = R0 / R_EARTH
    V0_scaled_in = V0 / V0_SCALE


    v_out = None

    _propagate = _kepler_j2_secular_propagate_torch if USE_J2_BASELINE else _kepler_propagate_torch

    T_max_scaled = (torch.full_like(t_scaled, T_gate_ref) / T0).clamp_min(1e-6)

    if compute_velocity:

        with torch.enable_grad():

            t_real = t_scaled * T0
            r_kepler = _propagate(x0_batch, u0_batch, t_real)
            r_kepler_scaled = r_kepler / R0

            inp = build_network_input(t_scaled * T0, x0_batch, u0_batch,
                                      r_kepler_scaled, t_scaled, R0, V0)
            r_nn_scaled = model.net(inp)
            gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0,
                                     torch.full_like(t_scaled, T_gate_ref))
            r_final_scaled = r_kepler_scaled + (gate_pow * r_nn_scaled)
            r = r_final_scaled * R0
            v_hat, _ = model._time_derivatives(r_final_scaled, t_scaled, need_acceleration=False, create_graph=False)
            v = v_hat * (R0 / T0)
        v_out = v.detach().cpu().numpy().astype(np.float64).reshape(S, npts, 3)
    else:

        with torch.no_grad():
            r_kepler = _propagate(x0_batch, u0_batch, t_batch)
            r_kepler_scaled = r_kepler / R0

            inp = build_network_input(t_scaled * T0, x0_batch, u0_batch,
                                      r_kepler_scaled, t_scaled, R0, V0)
            r_nn_scaled = model.net(inp)
            gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0,
                                     torch.full_like(t_scaled, T_gate_ref))
            r_final_scaled = r_kepler_scaled + (gate_pow * r_nn_scaled)
            r = r_final_scaled * R0

    r_out = r.detach().cpu().numpy().astype(np.float64).reshape(S, npts, 3)
    t_out = t_grid.astype(np.float64)

    return t_out, r_out, v_out

def predict_step_batch_chunked(model, x0_all_np, u0_all_np, T_total, npts=100,
                               satellite_chunk_size=64, Tseg_train=None,
                               show_progress=False):
    """Position-only inference for many satellites, in bounded chunks.

    Vectorises over satellites and time, then splits the flattened workload so
    the simulator (QNet) or GPU memory (Net) never sees the whole population at
    once. QNet splits each chunk further via quantum_batch_size.

    No autograd graph is built, so this is the correct fast path whenever only
    position is compared. Use predict_step_batch(compute_velocity=True) when
    velocity is genuinely needed.
    """
    model.net.eval()
    x0_all_np = np.asarray(x0_all_np, dtype=np.float64)
    u0_all_np = np.asarray(u0_all_np, dtype=np.float64)
    if x0_all_np.ndim != 2 or x0_all_np.shape[1] != 3:
        raise ValueError(f"x0_all_np must be [S,3], got {x0_all_np.shape}")
    if u0_all_np.shape != x0_all_np.shape:
        raise ValueError(f"u0_all_np {u0_all_np.shape} != x0_all_np {x0_all_np.shape}")
    if npts < 2:
        raise ValueError("npts must be at least 2")
    if satellite_chunk_size <= 0:
        raise ValueError("satellite_chunk_size must be positive")

    try:
        first = next(model.net.parameters())
        runtime_device, runtime_dtype = first.device, first.dtype
    except StopIteration:
        runtime_device, runtime_dtype = DEVICE, DTYPE

    # QNet keeps its state vector outside PyTorch and must stay on CPU.
    if model.net.__class__.__name__ == "QNet":
        want = model.net.recommended_torch_device
        if runtime_device.type != want.type:
            raise RuntimeError(
                f"QNet backend {model.net.quantum_backend!r} needs tensors on {want}, "
                f"but the model is on {runtime_device}. Build it with build_net(), "
                f"which places it correctly."
            )

    S = x0_all_np.shape[0]
    t_grid = np.linspace(0.0, T_total, npts, dtype=np.float64).reshape(-1, 1)
    r_out = np.empty((S, npts, 3), dtype=np.float64)

    for start in range(0, S, satellite_chunk_size):
        stop = min(start + satellite_chunk_size, S)
        C = stop - start
        x0 = torch.as_tensor(x0_all_np[start:stop], dtype=runtime_dtype, device=runtime_device)
        u0 = torch.as_tensor(u0_all_np[start:stop], dtype=runtime_dtype, device=runtime_device)
        t = torch.as_tensor(t_grid, dtype=runtime_dtype, device=runtime_device)

        x0_batch = x0.repeat_interleave(npts, dim=0)
        u0_batch = u0.repeat_interleave(npts, dim=0)
        t_batch = t.repeat(C, 1)

        with torch.no_grad():
            R0 = torch.linalg.vector_norm(x0_batch, dim=1, keepdim=True).clamp_min(1.0)
            V0 = torch.linalg.vector_norm(u0_batch, dim=1, keepdim=True).clamp_min(1.0)
            T0 = R0 / V0
            t_scaled = t_batch / T0

            r_base_scaled = model._kepler_baseline_scaled(x0_batch, u0_batch, t_batch, R0)
            inp = build_network_input(t_batch, x0_batch, u0_batch,
                                      r_base_scaled, t_scaled, R0, V0)
            residual = model.net(inp)

            tseg = (None if Tseg_train is None else
                    torch.full_like(t_batch, float(Tseg_train)))
            gate = _compute_gate(t_scaled, x0_batch, u0_batch, T0, tseg)
            r_chunk = (r_base_scaled + gate * residual) * R0

        r_out[start:stop] = r_chunk.reshape(C, npts, 3).detach().cpu().numpy().astype(np.float64)
        if show_progress:
            print(f"[predict] {stop}/{S} satellites", flush=True)

    return t_grid, r_out, None


def predict_autoregressive_batch(model, x0_all_np, u0_all_np, T_step, N_steps,
                                 npts_per_step=50, compute_velocity=False,
                                 Tseg_train=None):
    """
    Batched autoregressive prediction for S satellites at once.
    Stitches N_steps short-arc predictions together.

    Args:
        model:            GeneralistModel with loaded weights
        x0_all_np:        [S, 3] initial positions in meters
        u0_all_np:        [S, 3] initial velocities in meters/s
        T_step:           duration of each step in seconds
        N_steps:          number of autoregressive steps
        npts_per_step:    time points per step
        compute_velocity: if True, compute velocity via autograd (exact);
                          if False, use finite difference for next-step IC (faster)
        Tseg_train:       training window length (e.g. 80000.0).
                          Required for models trained with the per-sample gate
                          on long windows. Passes through to predict_step_batch
                          so the gate at predict time matches the gate at train time.
                          When None, each chunk is treated as its own
                          self-contained window (the original behavior).

    Returns:
        t_out:  [total_npts, 1]     time grid in seconds
        r_out:  [S, total_npts, 3]  predicted positions in meters
        v_out:  [S, total_npts, 3]  predicted velocities in m/s (only if compute_velocity=True, else None)
    """
    S = x0_all_np.shape[0]
    all_times = []
    all_positions = []
    all_velocities = [] if compute_velocity else None

    x0_c = x0_all_np.copy()
    u0_c = u0_all_np.copy()

    for i in range(N_steps):
        t_step, r_step, v_step = predict_step_batch(
            model, x0_c, u0_c, T_total=T_step, npts=npts_per_step,
            compute_velocity=compute_velocity,
            Tseg_train=Tseg_train)

        offset = i * T_step
        if i == 0:
            all_times.append(t_step + offset)
            all_positions.append(r_step)
            if compute_velocity:
                all_velocities.append(v_step)
        else:
            all_times.append(t_step[1:] + offset)
            all_positions.append(r_step[:, 1:, :])
            if compute_velocity:
                all_velocities.append(v_step[:, 1:, :])

        x0_c = r_step[:, -1, :].copy()
        if compute_velocity:
            u0_c = v_step[:, -1, :].copy()
        else:
            dt = t_step[-1, 0] - t_step[-2, 0]
            u0_c = (r_step[:, -1, :] - r_step[:, -2, :]) / dt

    t_out = np.concatenate(all_times, axis=0)
    r_out = np.concatenate(all_positions, axis=1)
    v_out = np.concatenate(all_velocities, axis=1) if compute_velocity else None

    return t_out, r_out, v_out

def predict_autoregressive_traj(model, x0_initial_np, u0_initial_np, T_step, N_steps,
                                npts_per_step=50, Tseg_train=None):
    """
    Predicts a LONG trajectory by stitching together N_steps short predictions.

    Tseg_train (optional): training window length. Pass the
    WINDOW_DURATION_SEC you used during training to make predict-time gate
    match train-time gate. When None, original behavior.
    """

    print(f"[Predict] Starting autoregressive prediction: {N_steps} steps of {T_step:.0f}s each...")

    all_times = []
    all_positions = []
    all_velocities = []

    x0_current = x0_initial_np.copy()
    u0_current = u0_initial_np.copy()

    for i in range(N_steps):
        t_step, r_step, v_step = predict_step(
            model,
            x0_current,
            u0_current,
            T_total=T_step,
            npts=npts_per_step,
            Tseg_train=Tseg_train,
        )

        current_time_offset = i * T_step

        if i == 0:
            all_times.append(t_step + current_time_offset)
            all_positions.append(r_step)
            all_velocities.append(v_step)
        else:
            all_times.append(t_step[1:] + current_time_offset)
            all_positions.append(r_step[1:])
            all_velocities.append(v_step[1:])

        x0_next = r_step[-1, :].copy()
        u0_next = v_step[-1, :].copy()

        x0_current = x0_next
        u0_current = u0_next

    print("[Predict] Autoregressive prediction finished.")

    t_final = np.vstack(all_times)
    r_final = np.vstack(all_positions)
    v_final = np.vstack(all_velocities)

    return t_final, r_final, v_final
