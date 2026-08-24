"""Decoder connection for tlsim.

Evaluates logical error rate (LER) at any of the three tlsim ambient noise
levels (``CodeCapacityNoise`` / ``PhenomenologicalNoise`` / ``CircuitLevelNoise``)
for a lossy circuit, using the same ``union_find.UFDecoder`` (ldpc Union-Find)
already used by the plain-stim P0-P5 pipeline in ``union_find.py`` /
``protocols.py`` — but fed from tsim's DEM of the *actual* loss/reload
circuit, rather than that pipeline's usual approximation (DEM of the
loss-free circuit, with LLR=0 patched in for detectors adjacent to a lost
qubit).

Why not reuse that approximation here: puncturing can make punctured
stabilizer generators anticommute, producing non-deterministic ("gauge")
detectors (see ``tlsim/usage_example.ipynb`` §4). When that happens, the
loss-free circuit's DEM has the wrong hypergraph structure for decoding
samples drawn from the punctured circuit — empirically this isn't just
inaccurate, it can make ``ldpc.UnionFindDecoder`` hang trying to satisfy a
syndrome pattern its (mismatched) PCM cannot explain. Building the DEM from
the punctured circuit itself via
``tsim`` (``circuit_builder.detector_error_model(..., allow_gauge_detectors=True)``)
sidesteps this: the PCM then matches exactly the circuit being sampled, gauge
detectors included, and no manual erasure-LLR patching is needed — the DEM's
own error probabilities already reflect the puncturing.

``LDUMode`` selects between oracle (noiseless) and physical (noisy) LDU
reload, mirroring ``protocols.evaluate_p1_oracle_reload`` /
``evaluate_p2_noisy_ldu_reload`` — swap ``kind`` to flip between them without
touching the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from union_find import UFDecoder, dem_to_pcm, probs_to_llrs

from tlsim.circuit_builder import (
    apply_loss_events,
    data_qubits,
    detector_error_model,
    get_qubit_coords,
    to_tsim,
)


@dataclass(frozen=True)
class LDUMode:
    """Oracle (noiseless) vs physical (noisy) LDU reload assumption.

    kind="noiseless" (default): oracle LDU — loss known instantly and
        exactly, reloaded ``reload_latency`` rounds later, no extra noise.
        Mirrors ``protocols.evaluate_p1_oracle_reload``.
    kind="noisy": physical circuit-based LDU — loss detected
        ``ldu_latency`` rounds after it occurs, reloaded ``reload_latency``
        rounds after that; the LDU circuit's own gate error (``p_ldu_gate``)
        is applied to every scanned data qubit at the detection round, plus
        back-propagation/reset noise (``p_reload``) at the reload round.
        Mirrors ``protocols.evaluate_p2_noisy_ldu_reload`` (arXiv:2405.10434).
    """

    kind: str = "noiseless"  # "noiseless" | "noisy"
    reload_latency: int = 1
    ldu_latency: int = 1
    p_ldu_gate: float = 0.0
    p_reload: float = 0.0
    ldu_type: str = "circuit"  # "circuit" | "swap" -> back-propagation factor


def _ldu_apply_kwargs(
    mode: LDUMode,
    loss_events: List[Tuple[int, int]],
    all_data_qubits: List[int],
    rounds: int,
) -> dict:
    """Return the ``apply_loss_events`` kwargs implied by an ``LDUMode``."""
    if mode.kind == "noiseless":
        reload_info = {
            int(q): min(int(r) + mode.reload_latency, rounds - 1)
            for r, q in loss_events
        }
        return dict(reload_info=reload_info, reload_basis="X", p_reload=0.0)

    if mode.kind != "noisy":
        raise ValueError(f"LDUMode.kind must be 'noiseless' or 'noisy', got {mode.kind!r}")

    prop_factor = {"circuit": 0.4, "swap": 0.2}.get(mode.ldu_type, 0.4)
    p_ldu_propagation = mode.p_ldu_gate * prop_factor
    min_loss_round = min(int(r) for r, _ in loss_events)
    ldu_detection_round = min(min_loss_round + mode.ldu_latency, rounds - 1)
    reload_info = {
        int(q): min(int(r) + mode.ldu_latency + mode.reload_latency, rounds - 1)
        for r, q in loss_events
    }
    return dict(
        reload_info=reload_info,
        reload_basis="X",
        p_reload=mode.p_reload,
        p_ldu_propagation=p_ldu_propagation,
        ldu_round=ldu_detection_round,
        ldu_target_qubits=all_data_qubits,
        p_ldu_noise=p_ldu_propagation,
    )


def build_ldu_circuit(
    c_flat: stim.Circuit,
    loss_events: List[Tuple[int, int]],
    rounds: int,
    mode: Optional[LDUMode],
) -> stim.Circuit:
    """Apply ``loss_events`` to ``c_flat`` under the given LDU reload mode.

    ``mode=None`` = P0 baseline: puncture only, no reload (erased until the
    end of the circuit). Otherwise dispatches to ``_ldu_apply_kwargs``.
    """
    if not loss_events:
        return c_flat
    if mode is None:
        return apply_loss_events(c_flat, loss_events, rounds)
    coords = get_qubit_coords(c_flat)
    all_dqs = data_qubits(coords)
    kwargs = _ldu_apply_kwargs(mode, loss_events, all_dqs, rounds)
    return apply_loss_events(c_flat, loss_events, rounds, **kwargs)


def decode_ler_with_ldu(
    c_flat: stim.Circuit,
    loss_events: List[Tuple[int, int]],
    rounds: int,
    mode: Optional[LDUMode],
    trials: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Evaluate LER for a lossy circuit at one ambient noise level.

    Builds the loss/reload circuit for ``mode``, gets *its own* DEM via tsim
    (``allow_gauge_detectors=True`` — robust to the non-deterministic
    detectors puncturing can create), decodes with ``UFDecoder`` using that
    DEM's own error probabilities (no manual erasure-LLR patching needed —
    see module docstring), and samples from the same tsim circuit with
    ``use_detector_reference_sample=True`` so results are the flip-relative
    detection events the DEM's probabilities describe.

    Parameters
    ----------
    c_flat : stim.Circuit
        Flattened, loss-free circuit at the desired ambient noise level
        (the second return value of
        ``tlsim.circuit_builder.build_surface_code_circuit``).
    loss_events : list of (round, qubit)
        True loss events.
    rounds : int
        Total syndrome rounds.
    mode : LDUMode or None
        Reload assumption. ``None`` = P0 baseline (no reload).
    trials : int
        Number of sampled shots.
    seed : int, optional
        Seed for tsim's detector sampler.

    Returns
    -------
    dict with keys: ldu_mode, logical_failure_rate, syndrome_intensity,
                    n_error_mechanisms
    """
    c_sample = build_ldu_circuit(c_flat, loss_events, rounds, mode)
    tc = to_tsim(c_sample)
    dem = detector_error_model(tc, allow_gauge_detectors=True)
    pcm, log_obs, probs = dem_to_pcm(dem)
    if pcm.shape[1] == 0:
        raise ValueError(
            "DEM has 0 error mechanisms (PCM is empty). Use a non-zero "
            "ambient noise level (e.g. p_pauli=1e-6) so the UFDecoder has a "
            "non-trivial PCM to decode against."
        )
    decoder = UFDecoder(pcm, log_obs)
    llrs = probs_to_llrs(probs)

    sampler = tc.compile_detector_sampler(seed=seed)
    det_ev, obs = sampler.sample(
        shots=trials,
        separate_observables=True,
        use_detector_reference_sample=True,
        use_observable_reference_sample=True,
    )
    det_ev = det_ev.astype(np.uint8)
    obs = obs.astype(np.uint8).reshape(-1)

    failures = decoder.decode_batch(det_ev, llrs=llrs) ^ obs.astype(bool)

    return {
        "ldu_mode": "none" if mode is None else mode.kind,
        "logical_failure_rate": float(failures.mean()),
        "syndrome_intensity": float(det_ev.sum(axis=1).mean()),
        "n_error_mechanisms": int(pcm.shape[1]),
    }
