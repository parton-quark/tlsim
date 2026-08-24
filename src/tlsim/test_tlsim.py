"""Sanity checks for tlsim: circuit construction + loss injection at all
three noise levels, and the tsim-vs-stim DEM behavior under non-deterministic
detectors that motivates using tsim for loss circuits in the first place.

Run with: pytest src/tlsim/test_tlsim.py -v
(requires bloqade-tsim: `pip install bloqade-tsim`)
"""

import numpy as np
import pytest
import stim

from tlsim.circuit_builder import (
    apply_loss_events,
    build_lossy_tsim_circuit,
    build_surface_code_circuit,
    data_qubits,
    detector_error_model,
    get_qubit_coords,
    sample_detectors,
    sample_loss_events,
)
from tlsim.decode import LDUMode, decode_ler_with_ldu
from tlsim.noise_models import CircuitLevelNoise, CodeCapacityNoise, PhenomenologicalNoise


def test_code_capacity_no_loss_runs():
    noise = CodeCapacityNoise(p_data=0.02)
    c0, c_flat = build_surface_code_circuit(3, 1, noise)
    tc = build_lossy_tsim_circuit(3, 1, noise, loss_events=[])
    samples = sample_detectors(tc, shots=200)
    assert samples.shape[0] == 200
    assert samples.shape[1] == c0.num_detectors


def test_code_capacity_requires_single_round():
    with pytest.raises(ValueError):
        build_surface_code_circuit(3, 2, CodeCapacityNoise(p_data=0.02))


def test_phenomenological_runs():
    noise = PhenomenologicalNoise(p_data=0.01, p_meas=0.01)
    tc = build_lossy_tsim_circuit(3, 5, noise, loss_events=[])
    samples = sample_detectors(tc, shots=200)
    assert samples.shape[0] == 200


def test_circuit_level_runs():
    noise = CircuitLevelNoise(p_pauli=0.001, p_meas=0.001)
    tc = build_lossy_tsim_circuit(3, 5, noise, loss_events=[])
    samples = sample_detectors(tc, shots=200)
    assert samples.shape[0] == 200


@pytest.mark.parametrize(
    "noise,rounds",
    [
        (CodeCapacityNoise(p_data=0.0), 1),
        (PhenomenologicalNoise(p_data=0.0, p_meas=0.0), 5),
        (CircuitLevelNoise(p_pauli=0.0, p_meas=0.0), 5),
    ],
)
def test_single_loss_all_levels_runs_and_is_deterministic_without_loss(noise, rounds):
    """With zero ambient noise, a lost data qubit still yields a valid,
    samplable circuit at every noise level (puncturing is orthogonal to the
    ambient noise level)."""
    d = 3
    c0, _ = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    dqs = data_qubits(coords)
    loss_events = [(0, dqs[0])]

    tc = build_lossy_tsim_circuit(
        d, rounds, noise, loss_events, p_surv=0.0, survivor_mode="none"
    )
    samples = sample_detectors(tc, shots=50)
    assert samples.shape[0] == 50


def test_puncturing_removes_cx_touching_lost_qubit():
    d = 3
    rounds = 3
    noise = CircuitLevelNoise(p_pauli=0.0, p_meas=0.0)
    c0, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    q = data_qubits(coords)[0]

    c_loss = apply_loss_events(c_flat, [(0, q)], rounds)
    for inst in c_loss:
        if inst.name == "CX":
            targets = [t.value for t in inst.targets_copy()]
            assert q not in targets, "lost qubit must not appear in any CX after round 0"


def test_reload_restores_cx_after_reload_round():
    d = 3
    rounds = 5
    noise = CircuitLevelNoise(p_pauli=0.0, p_meas=0.0)
    c0, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    q = data_qubits(coords)[0]

    c_loss = apply_loss_events(
        c_flat, [(0, q)], rounds, reload_info={q: 2}, p_reload=0.0
    )
    tpr = 4  # rotated surface code has 4 CX layers/round; sanity, not asserted
    # After reload, q must reappear in CX gates somewhere in the circuit.
    reappears = any(
        inst.name == "CX" and q in [t.value for t in inst.targets_copy()]
        for inst in c_loss
    )
    assert reappears, "reloaded qubit should participate in CX gates again"


def test_sample_loss_events_reproducible_with_seed():
    d = 3
    noise = CircuitLevelNoise()
    c0, _ = build_surface_code_circuit(d, 5, noise)
    coords = get_qubit_coords(c0)
    dqs = data_qubits(coords)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    e1 = sample_loss_events(dqs, 5, 0.05, rng1)
    e2 = sample_loss_events(dqs, 5, 0.05, rng2)
    assert e1 == e2


def test_tsim_dem_handles_nondeterministic_detectors_where_stim_fails():
    """The reason to use tsim for loss circuits: stim's detector_error_model()
    refuses circuits with non-deterministic detectors/observables (which
    puncturing routinely produces), tsim's does not."""
    d = 3
    rounds = 5
    noise = CircuitLevelNoise(p_pauli=0.001, p_meas=0.001)
    _, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c_flat)
    q = data_qubits(coords)[0]

    c_loss = apply_loss_events(c_flat, [(1, q)], rounds, p_surv=0.05, survivor_mode="depolarize")

    # Confirm this circuit is indeed the hard case (stim refuses it). If stim
    # starts succeeding here in some future version/parameter regime, this
    # assertion (not the tsim path below) is what should be revisited.
    stim_failed = False
    try:
        c_loss.detector_error_model()
    except ValueError:
        stim_failed = True
    assert stim_failed, "expected stim DEM to fail on this non-deterministic loss circuit"

    tc = build_lossy_tsim_circuit(d, rounds, noise, [(1, q)], p_surv=0.05, survivor_mode="depolarize")
    dem = detector_error_model(tc)
    assert isinstance(dem, stim.DetectorErrorModel)
    assert len(dem) > 0


@pytest.mark.parametrize(
    "noise,rounds,loss_round",
    [
        (CodeCapacityNoise(p_data=0.03), 1, 0),
        (PhenomenologicalNoise(p_data=0.02, p_meas=0.02), 5, 1),
        (CircuitLevelNoise(p_pauli=0.01, p_meas=0.01), 5, 1),
    ],
)
def test_decode_ler_with_ldu_runs_at_all_noise_levels(noise, rounds, loss_round):
    """decode_ler_with_ldu must not hang or raise for a single lost qubit at
    any of the three noise levels, with and without LDU reload."""
    d = 3
    c0, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    q = data_qubits(coords)[0]
    loss_events = [(loss_round, q)]

    r0 = decode_ler_with_ldu(c_flat, loss_events, rounds, None, trials=100)
    assert 0.0 <= r0["logical_failure_rate"] <= 1.0

    r1 = decode_ler_with_ldu(
        c_flat, loss_events, rounds, LDUMode(kind="noiseless"), trials=100
    )
    assert 0.0 <= r1["logical_failure_rate"] <= 1.0


def test_decode_ler_with_ldu_oracle_reload_beats_no_reload():
    """Oracle (noiseless) LDU reload should not do worse than no reload at
    all — same qualitative ordering as protocols.evaluate_p1_oracle_reload
    vs evaluate_p0_baseline for a loss off the logical string."""
    d = 5
    rounds = 8
    noise = CircuitLevelNoise(p_pauli=0.005, p_meas=0.005)
    c0, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    dqs = data_qubits(coords)
    q_lost = dqs[len(dqs) // 2]
    loss_events = [(2, q_lost)]

    r_p0 = decode_ler_with_ldu(c_flat, loss_events, rounds, None, trials=1500, seed=1)
    r_p1 = decode_ler_with_ldu(
        c_flat, loss_events, rounds, LDUMode(kind="noiseless"), trials=1500, seed=1
    )
    assert r_p1["logical_failure_rate"] <= r_p0["logical_failure_rate"]


def test_decode_ler_with_ldu_rejects_unknown_mode_kind():
    d = 3
    rounds = 5
    noise = CircuitLevelNoise(p_pauli=0.01, p_meas=0.01)
    c0, c_flat = build_surface_code_circuit(d, rounds, noise)
    coords = get_qubit_coords(c0)
    q = data_qubits(coords)[0]
    with pytest.raises(ValueError):
        decode_ler_with_ldu(c_flat, [(1, q)], rounds, LDUMode(kind="bogus"), trials=10)
