"""Pre-generalization implementation retained for historical reference."""

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

T_SCALED_MAX = round(80000.0 / 907.0, 1)
V0_SCALE = 7500.0
KEPLER_DTYPE = torch.float64

USE_J2_BASELINE = False

USE_LAT_FEATURES = False

USE_OSC2MEAN = False

GATE_POWER = 3.0

GATE_KIND    = 'legacy'
GATE_TAU_DIV = 3.0

USE_ORBIT_FEATURES = False
_N_ORBIT_FEATURES  = 5

_N_LAT_FEATURES = 2

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
    """Five extra inputs: [cos_i, sin2_i, ecc, sin_u, cos_u].

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

    return torch.cat([cos_i, sin2_i, ecc, sin_u, cos_u], dim=1)

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

        feat_list = [t_scaled, x0_scaled, u0_scaled, R0_scaled_in, V0_scaled_in]
        if USE_LAT_FEATURES:
            lat_feats = _lat_features(r_kepler_scaled)
            feat_list.append(lat_feats)
        if USE_ORBIT_FEATURES:
            feat_list.append(_orbit_features(x0_batch, u0_batch, r_kepler_scaled))

        inp = torch.cat(feat_list, dim=1)
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

            scale = (torch.norm(a_true_pde, dim=1, keepdim=True) + torch.norm(a_pde_nn, dim=1, keepdim=True) + 1e-9)
            residual_pde = (a_pde_nn - a_true_pde) / scale
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

        feat_list = [t_scaled, x0_scaled, u0_scaled, R0_scaled_in, V0_scaled_in]
        if USE_LAT_FEATURES:
            feat_list.append(_lat_features(r_kepler_scaled))
        if USE_ORBIT_FEATURES:
            feat_list.append(_orbit_features(x0_batch, u0_batch, r_kepler_scaled))
        inp = torch.cat(feat_list, dim=1)
        r_nn_scaled = model.net(inp)

        gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0,
                                 torch.full_like(t_scaled, T_gate_ref))
        r_final_scaled = r_kepler_scaled + (gate_pow * r_nn_scaled)
        r = r_final_scaled * R0

    with torch.enable_grad():
        t_scaled_grad = t_torch / T0

        r_kepler_grad = _propagate(x0_batch, u0_batch, t_scaled_grad * T0)
        r_kepler_scaled_grad = r_kepler_grad / R0

        feat_list_grad = [t_scaled_grad, x0_scaled, u0_scaled, R0_scaled_in, V0_scaled_in]
        if USE_LAT_FEATURES:
            feat_list_grad.append(_lat_features(r_kepler_scaled_grad))
        if USE_ORBIT_FEATURES:
            feat_list_grad.append(_orbit_features(x0_batch, u0_batch, r_kepler_scaled_grad))
        inp_grad = torch.cat(feat_list_grad, dim=1)
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

    _base_feats = [t_scaled, x0_scaled, u0_scaled, R0_scaled_in, V0_scaled_in]

    v_out = None

    _propagate = _kepler_j2_secular_propagate_torch if USE_J2_BASELINE else _kepler_propagate_torch

    T_max_scaled = (torch.full_like(t_scaled, T_gate_ref) / T0).clamp_min(1e-6)

    if compute_velocity:

        with torch.enable_grad():

            t_real = t_scaled * T0
            r_kepler = _propagate(x0_batch, u0_batch, t_real)
            r_kepler_scaled = r_kepler / R0

            _feats = list(_base_feats)
            if USE_LAT_FEATURES:
                _feats.append(_lat_features(r_kepler_scaled))
            if USE_ORBIT_FEATURES:
                _feats.append(_orbit_features(x0_batch, u0_batch, r_kepler_scaled))
            inp = torch.cat(_feats, dim=1)
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

            _feats = list(_base_feats)
            if USE_LAT_FEATURES:
                _feats.append(_lat_features(r_kepler_scaled))
            if USE_ORBIT_FEATURES:
                _feats.append(_orbit_features(x0_batch, u0_batch, r_kepler_scaled))
            inp = torch.cat(_feats, dim=1)
            r_nn_scaled = model.net(inp)
            gate_pow = _compute_gate(t_scaled, x0_batch, u0_batch, T0,
                                     torch.full_like(t_scaled, T_gate_ref))
            r_final_scaled = r_kepler_scaled + (gate_pow * r_nn_scaled)
            r = r_final_scaled * R0

    r_out = r.detach().cpu().numpy().astype(np.float64).reshape(S, npts, 3)
    t_out = t_grid.astype(np.float64)

    return t_out, r_out, v_out

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
