"""VERIFY -- run after any edit to pinn_lib_kepler.py, before launching a run.

Covers: regimes, Fourier embedding, harmonic recursion, feature/dim agreement
at every call site, hard constraint at t=0, physics guards, and the quantum
layer construction and forward pass.

    python3 verify_lib.py           # skips quantum if pennylane is absent
Exit 0 means everything passed.
"""
import math, sys
import numpy as np, torch
import pinn_lib_kepler as lib

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok: FAIL.append(name)

def state(a, e, inc, O=0.7, w=1.1, nu=0.3):
    mu = lib.MU_EARTH; p = a*(1-e*e); rm = p/(1+e*math.cos(nu))
    rp = np.array([rm*math.cos(nu), rm*math.sin(nu), 0.0])
    vp = math.sqrt(mu/p)*np.array([-math.sin(nu), e+math.cos(nu), 0.0])
    cO,sO,ci,si,cw,sw = math.cos(O),math.sin(O),math.cos(inc),math.sin(inc),math.cos(w),math.sin(w)
    R = (np.array([[cO,-sO,0],[sO,cO,0],[0,0,1]])
         @ np.array([[1,0,0],[0,ci,-si],[0,si,ci]])
         @ np.array([[cw,-sw,0],[sw,cw,0],[0,0,1]]))
    return R@rp, R@vp

def batch(inc_deg, n=32, alt=700e3, e=0.005, dev=None):
    r0, v0 = state(lib.R_EARTH+alt, e, math.radians(inc_deg))
    d = dev or lib.DEVICE
    x0 = torch.tensor(np.repeat(r0[None],n,0), dtype=lib.DTYPE, device=d)
    u0 = torch.tensor(np.repeat(v0[None],n,0), dtype=lib.DTYPE, device=d)
    t  = torch.linspace(0., 80000., n, dtype=lib.DTYPE, device=d).reshape(-1,1)
    return r0, v0, x0, u0, t

def reset():
    lib.set_regime("LEO"); lib.USE_J2_BASELINE=True; lib.USE_OSC2MEAN=True
    lib.USE_LAT_FEATURES=False; lib.USE_ORBIT_FEATURES=False
    lib.USE_ORBIT_PHASE_FEATURES=True; lib.ORBIT_N_HARMONICS=1; lib.USE_FOURIER_TIME=False
    lib.FOURIER_N_FREQS=4; lib.FOURIER_SPACING='integer'
    lib.GATE_KIND='tanh'; lib.GATE_TAU_DIV=3.0; lib.GATE_POWER=2.0
    lib.PDE_NORM='total'; lib.PDE_GATE_WEIGHTED=False; lib.PDE_SUBTRACT_KEPLER=False
    lib.USE_J2=False
    lib.USE_QUANTUM_INPUT=False; lib.USE_QUANTUM_HIDDEN=False
    lib.QUANTUM_N_QUBITS=9; lib.QUANTUM_DEPTH=4; lib.QUANTUM_HIDDEN_POSITION=1
    lib.FOURIER_MAX_FREQ=50.0; lib.FOURIER_LEARNABLE=False
    lib.FOURIER_EMBEDDING=None

print("1. defaults are backward compatible")
import importlib; fresh = importlib.reload(lib)
check("REGIME defaults LEO", fresh.REGIME == "LEO")
check("GATE_KIND defaults legacy", fresh.GATE_KIND == "legacy")
check("ORBIT_N_HARMONICS defaults 1", fresh.ORBIT_N_HARMONICS == 1)
check("USE_ORBIT_PHASE_FEATURES defaults True", fresh.USE_ORBIT_PHASE_FEATURES is True)
check("USE_FOURIER_TIME defaults False", fresh.USE_FOURIER_TIME is False)
check("USE_QUANTUM_INPUT defaults False", fresh.USE_QUANTUM_INPUT is False)
check("USE_QUANTUM_HIDDEN defaults False", fresh.USE_QUANTUM_HIDDEN is False)
check("base input_dim is 9", fresh.input_dim() == 9, f"got {fresh.input_dim()}")
reset()

print("\n2. regimes")
for name, v0 in (("LEO",7500.0), ("MEO",3870.0), ("GEO",3100.0)):
    out = lib.set_regime(name)
    check(f"{name} V0_SCALE", abs(lib.V0_SCALE-v0) < 1e-9, f"{lib.V0_SCALE}")
check("GEO window override", abs(lib.set_regime("GEO", 10000.0)["T_SCALED_MAX"]-0.7) < 1e-6)
try:
    lib.set_regime("SSO"); check("rejects unknown regime", False)
except ValueError:
    check("rejects unknown regime", True)
reset()

print("\n3. input_dim tracks every flag combination")
cases = [((False,1,False,0), 9), ((True,1,False,0), 14), ((True,3,False,0), 18),
         ((True,4,False,0), 20), ((True,4,True,0), 22), ((False,1,False,4), 17),
         ((True,4,False,4), 28), ((False,1,True,4), 19)]
for (orb,h,lat,fk), want in cases:
    lib.USE_ORBIT_FEATURES=orb; lib.ORBIT_N_HARMONICS=h
    lib.USE_LAT_FEATURES=lat; lib.USE_FOURIER_TIME=fk>0; lib.FOURIER_N_FREQS=max(fk,1)
    got = lib.input_dim()
    check(f"orbit={orb} h={h} lat={lat} fourier={fk} -> {want}", got==want, f"got {got}")
reset()

lib.USE_ORBIT_FEATURES=True; lib.USE_ORBIT_PHASE_FEATURES=False
lib.USE_FOURIER_TIME=True; lib.FOURIER_N_FREQS=4
check("static orbit + Fourier-4 input_dim is 20", lib.input_dim() == 20,
      f"got {lib.input_dim()}")
_,_,x0_static,u0_static,_ = batch(51.6, n=4)
static_features = lib._orbit_features(
    x0_static, u0_static, x0_static / torch.norm(x0_static, dim=1, keepdim=True))
check("static orbit block is [inclination descriptors, eccentricity] only",
      static_features.shape == (4, 3), f"got {tuple(static_features.shape)}")
static_net = lib.attach_fourier(lib.build_net([lib.input_dim(), 16, 3]))
static_model = lib.GeneralistModel(static_net, W_DATA=1.0, W_PDE=0.0)
r0_static, v0_static, _, _, _ = batch(51.6, n=4)
try:
    _, static_prediction, _ = lib.predict_step(
        static_model, r0_static, v0_static, T_total=1000.0, npts=4,
        Tseg_train=40000.0)
    check("static orbit + Fourier path predicts",
          static_prediction.shape == (4, 3) and np.isfinite(static_prediction).all())
except Exception as exc:
    check("static orbit + Fourier path predicts", False,
          f"{type(exc).__name__}: {exc}")
reset()

print("\n4. harmonic recursion vs direct trigonometry")
_,_,x0,u0,_ = batch(51.6)
lib.ORBIT_N_HARMONICS=4
f = lib._orbit_features(x0, u0, x0/torch.norm(x0,dim=1,keepdim=True))
check("N=4 gives 11 columns", f.shape[1]==11, f"got {f.shape[1]}")
u = torch.atan2(f[:,3], f[:,4])
for k in (2,3,4):
    sc, cc = f[:,3+2*(k-1)], f[:,4+2*(k-1)]
    check(f"sin {k}u", torch.allclose(sc, torch.sin(k*u), atol=1e-4),
          f"max err {float((sc-torch.sin(k*u)).abs().max()):.2e}")
    check(f"cos {k}u", torch.allclose(cc, torch.cos(k*u), atol=1e-4),
          f"max err {float((cc-torch.cos(k*u)).abs().max()):.2e}")
reset()

print("\n5. Fourier embedding")
_,_,x0,u0,t = batch(51.6, n=64)
lib.USE_FOURIER_TIME=True; lib.FOURIER_N_FREQS=4; lib.FOURIER_SPACING='integer'
fe = lib._fourier_time_features(t, x0, u0)
check("integer spacing gives 2K columns", fe.shape[1]==8, f"got {fe.shape[1]}")
check("all finite", bool(torch.isfinite(fe).all()))
# Column layout is GROUPED, matching the reference cat([t, sin(z), cos(z)]):
#   [sin w1, sin w2, .., sin wK, cos w1, .., cos wK]
# not interleaved. Checkpoints depend on this order, so it is asserted.
K = 4
check("sin^2+cos^2=1 per frequency (grouped layout)",
      all(torch.allclose(fe[:, i]**2 + fe[:, K + i]**2,
                         torch.ones(64, dtype=lib.DTYPE), atol=1e-4)
          for i in range(K)))
check("first K columns are sines (zero at t=0)",
      torch.allclose(lib._fourier_time_features(torch.zeros(64,1,dtype=lib.DTYPE), x0, u0)[:, :K],
                     torch.zeros(64, K, dtype=lib.DTYPE), atol=1e-6))
check("last K columns are cosines (one at t=0)",
      torch.allclose(lib._fourier_time_features(torch.zeros(64,1,dtype=lib.DTYPE), x0, u0)[:, K:],
                     torch.ones(64, K, dtype=lib.DTYPE), atol=1e-6))
T_orb = lib._orbital_period_torch(x0,u0)
check("period matches one revolution",
      torch.allclose(lib._fourier_time_features(T_orb, x0, u0)[:,0],
                     torch.zeros(64,dtype=lib.DTYPE), atol=1e-3),
      "sin(2pi) = 0 at t = T_orb")
lib.FOURIER_SPACING='log'
check("log spacing frequencies are 1,2,4,8", lib._fourier_frequencies()==[1.,2.,4.,8.])
lib.FOURIER_SPACING='geometric'; lib.FOURIER_MAX_FREQ=50.0
gf = lib._fourier_frequencies()
check("geometric spans 1..max_freq (reference form)",
      abs(gf[0]-1.0)<1e-5 and abs(gf[-1]-50.0)<1e-3, f"{[round(x,3) for x in gf]}")
try:
    lib.FourierEmbedding(4, spacing='geometric', max_freq=1.0)
    check("geometric rejects max_freq=1 (reference's degenerate default)", False)
except ValueError:
    check("geometric rejects max_freq=1 (reference's degenerate default)", True)
emb = lib.FourierEmbedding(4, spacing='integer', learnable=True)
check("learnable frequencies are Parameters",
      any(p.requires_grad for p in emb.parameters()), "reference used register_buffer")
check("fixed frequencies are buffers",
      len(list(lib.FourierEmbedding(4).parameters()))==0)
check("out_dim is 2K (raw t supplied separately)", lib.FourierEmbedding(4).out_dim==8)
lib.FOURIER_SPACING='integer'; lib.FOURIER_MAX_FREQ=50.0
lib.FOURIER_SPACING='integer'
check("integer spacing frequencies are 1,2,3,4", lib._fourier_frequencies()==[1.,2.,3.,4.])
reset()

print("\n6. equatorial guard across inclination")
lib.USE_ORBIT_FEATURES=True; lib.ORBIT_N_HARMONICS=4
for label, inc in [("51.6",51.6), ("near-equatorial 0.05",0.05), ("polar 90",90.0),
                   ("SSO 97.4",97.4), ("retro 110",110.0)]:
    _,_,x0,u0,_ = batch(inc)
    f = lib._orbit_features(x0,u0,x0/torch.norm(x0,dim=1,keepdim=True))
    ok = torch.isfinite(f).all() and abs(float(f[0,0])-math.cos(math.radians(inc)))<2e-3
    check(f"inc {label}", bool(ok), f"cos i={float(f[0,0]):+.4f} finite={bool(torch.isfinite(f).all())}")
reset()

print("\n7. hard constraint and full forward, every feature on")
lib.USE_ORBIT_FEATURES=True; lib.ORBIT_N_HARMONICS=4
lib.USE_LAT_FEATURES=True; lib.USE_FOURIER_TIME=True; lib.FOURIER_N_FREQS=4
dim = lib.input_dim()
check("combined dim is 30", dim==30, f"got {dim}")
torch.manual_seed(0)
net = lib.build_net([dim,32,32,3])
model = lib.GeneralistModel(net, W_DATA=1.0, W_PDE=0.0)
r0,v0,x0,u0,t = batch(51.6, n=8)
t0 = torch.zeros(8,1,dtype=lib.DTYPE,device=lib.DEVICE).requires_grad_(True)
tseg = torch.full_like(t0, 80000.)
r_nn, v_nn, _, _ = model.get_scaled_state_rv(t0, x0, u0, Tseg_batch=tseg)
dr = float(torch.norm(r_nn-x0,dim=1).max().detach())
check("r(0) = r0", dr < 1.0, f"max |dr| = {dr:.3e} m")
g0 = lib._compute_gate(torch.zeros(4,1,dtype=lib.DTYPE,device=lib.DEVICE), x0[:4], u0[:4],
                       torch.ones(4,1,dtype=lib.DTYPE,device=lib.DEVICE),
                       torch.full((4,1),80000.,dtype=lib.DTYPE,device=lib.DEVICE))
check("gate(0) = 0 exactly", float(g0.abs().max())==0.0)

print("\n8. all inference call sites accept the same width")
tt = t.clone().requires_grad_(True)
rs,_,_,_,_ = model._forward_scaled_position(tt, x0, u0, Tseg_batch=torch.full_like(tt,80000.))
check("_forward_scaled_position", rs.shape==(8,3), str(tuple(rs.shape)))
try:
    _,rp,_ = lib.predict_step(model, r0, v0, T_total=8000., npts=16, Tseg_train=80000.)
    check("predict_step", np.isfinite(rp).all() and rp.shape==(16,3), str(rp.shape))
except Exception as e:
    check("predict_step", False, f"{type(e).__name__}: {e}")
try:
    _,rb,_ = lib.predict_step_batch(model, r0[None], v0[None], T_total=8000., npts=16)
    check("predict_step_batch", np.isfinite(rb).all(), str(np.asarray(rb).shape))
except Exception as e:
    check("predict_step_batch", False, f"{type(e).__name__}: {e}")
try:
    _,rc,_ = lib.predict_step_batch_chunked(model, np.repeat(r0[None],5,0),
                                            np.repeat(v0[None],5,0), T_total=8000.,
                                            npts=16, satellite_chunk_size=2)
    check("predict_step_batch_chunked", np.isfinite(rc).all() and rc.shape==(5,16,3), str(rc.shape))
except Exception as e:
    check("predict_step_batch_chunked", False, f"{type(e).__name__}: {e}")

print("\n9. physics branch")
lib.USE_J2 = True
r_test = x0[:4].clone()
a_full = lib.earth_gravity_acc_torch(r_test)
lib.USE_J2 = False
a_two = lib.earth_gravity_acc_torch(r_test)
ratio = float((torch.norm(a_full-a_two,dim=1)/torch.norm(a_two,dim=1)).mean())
check("J2 is ~1e-3 of central when USE_J2=True", 5e-4 < ratio < 5e-3, f"ratio {ratio:.2e}")
check("USE_J2=False gives pure two-body",
      torch.allclose(a_two, -lib.MU_EARTH*r_test/torch.norm(r_test,dim=1,keepdim=True)**3, rtol=1e-4))
lib.PDE_NORM='j2'; lib.USE_J2=False
empty1 = torch.empty(0,1,dtype=lib.DTYPE,device=lib.DEVICE)
empty3 = torch.empty(0,3,dtype=lib.DTYPE,device=lib.DEVICE)
tp = torch.full((4,1),1000.,dtype=lib.DTYPE,device=lib.DEVICE).requires_grad_(True)
env = (torch.zeros(1,3,dtype=lib.DTYPE,device=lib.DEVICE),)*2 + \
      (torch.tensor(0.,dtype=lib.DTYPE,device=lib.DEVICE),)*6 + \
      (torch.zeros(1,3,dtype=lib.DTYPE,device=lib.DEVICE),)*2
try:
    model.epoch_loss(empty1,empty3,empty3,empty3,empty3, tp,x0[:4],u0[:4], *env,
                     Tseg_data_batch=None, Tseg_pde_batch=torch.full_like(tp,80000.))
    check("PDE_NORM='j2' raises when USE_J2=False", False, "no exception")
except ValueError as e:
    check("PDE_NORM='j2' raises when USE_J2=False", "requires USE_J2=True" in str(e))
lib.PDE_NORM='total'
reset()

print("\n10. quantum variants (QAPINN)")
try:
    import pennylane  # noqa: F401
    have_pl = True
except ImportError:
    have_pl = False
if not have_pl:
    print("  [SKIP] pennylane not installed; quantum path untested here")
else:
    lib.USE_ORBIT_FEATURES=True; lib.ORBIT_N_HARMONICS=4
    lib.QUANTUM_N_QUBITS=6; lib.QUANTUM_DEPTH=2
    dim = lib.input_dim(); rw = lib.quantum_readout_width(6)
    check("readout width is 2**(n_qubits-1)", rw==32, f"got {rw}")

    lib.USE_QUANTUM_INPUT=True; lib.USE_QUANTUM_HIDDEN=False
    qa = lib.build_net([dim, rw, 16, 3])
    check("variant A builds", qa.variant=="A_quantum_input", qa.variant)
    check("A: circuit at position 0", qa.quantum_position==0)
    check("A: projection inserted for dim != n_qubits", qa.input_projection is not None)
    oa = qa(torch.randn(6, dim, dtype=lib.DTYPE))
    check("A: forward [B,3] finite", oa.shape==(6,3) and bool(torch.isfinite(oa).all()))
    oa.pow(2).mean().backward()
    ga = qa.linear[0].weights.grad
    check("A: backprop reaches quantum weights", ga is not None and float(ga.norm())>0,
          f"|g| = {float(ga.norm()):.3e}")

    lib.USE_QUANTUM_INPUT=False; lib.USE_QUANTUM_HIDDEN=True; lib.QUANTUM_HIDDEN_POSITION=1
    qb = lib.build_net([dim, 16, rw, 16, 3])
    check("variant B builds", qb.variant=="B_quantum_hidden", qb.variant)
    check("B: circuit at position 1", qb.quantum_position==1)
    check("B: layer 0 is classical", isinstance(qb.linear[0], torch.nn.Linear))
    ob = qb(torch.randn(6, dim, dtype=lib.DTYPE))
    check("B: forward [B,3] finite", ob.shape==(6,3) and bool(torch.isfinite(ob).all()))
    ob.pow(2).mean().backward()
    gb = qb.linear[1].weights.grad
    check("B: backprop reaches quantum weights", gb is not None and float(gb.norm())>0,
          f"|g| = {float(gb.norm()):.3e}")

    check("batch broadcasts (no per-sample loop)",
          qa(torch.randn(32, dim, dtype=lib.DTYPE)).shape==(32,3))
    check("checkpoint key prefix matches Net ('linear.')",
          all(k.startswith(("linear.", "input_projection.")) for k in qa.state_dict()))

    lib.USE_QUANTUM_INPUT=True
    try:
        lib.build_net([dim, rw, 16, 3]); check("both variants on is rejected", False)
    except ValueError:
        check("both variants on is rejected", True)
    lib.USE_QUANTUM_INPUT=True; lib.USE_QUANTUM_HIDDEN=False
    try:
        lib.build_net([dim, rw+1, 16, 3]); check("wrong readout width rejected", False)
    except ValueError:
        check("wrong readout width rejected", True)
    reset()

print("\n" + "="*62)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
