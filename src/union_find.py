"""
union_find.py

Active Union-Find decoder for the rotated surface code (P0–P5 evaluation).

UFDecoder — wraps ldpc.UnionFindDecoder (matrix mode) with per-call LLRs.
   - Works directly on the DEM PCM; handles all DEM hyperedges correctly.
   - Erasure: mechanisms adjacent to lost qubits receive LLR = 0 (p_eff = 0.5).
   - Correctly suppresses Pauli errors with code distance d.
   - decode_and_evaluate caches PCM + UFDecoder per circuit to avoid redundant rebuilds.

Legacy decoders (PeelingErasureDecoder, SpaceTimePeelingDecoder) have been
moved to old_decoder.py and are not used in the main evaluation pipeline.
"""

import math
import numpy as np
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# DEM → PCM + metadata
# ---------------------------------------------------------------------------

def infer_rounds_from_circuit(c_flat) -> int:
    """
    Infer the number of syndrome rounds from a flattened circuit.

    Calls build_detector_map with a large sentinel (so ticks_per_round is
    computed correctly), then returns max_round_index + 1.
    """
    det_map = build_detector_map(c_flat, rounds=1000)
    if not det_map:
        return 1
    return max(r for _, r in det_map.values()) + 1


def dem_to_pcm(dem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a Stim DetectorErrorModel to a binary PCM and associated vectors.

    Parameters
    ----------
    dem : stim.DetectorErrorModel

    Returns
    -------
    pcm       : uint8 array (n_detectors, n_errors)
    log_obs   : uint8 array (n_errors,)  — 1 if mechanism flips logical L0
    probs     : float64 array (n_errors,) — error probabilities from DEM
    """
    errors = []
    for inst in dem.flattened():
        if inst.type == "error":
            dets = [t.val for t in inst.targets_copy()
                    if t.is_relative_detector_id()]
            flips_log = any(t.is_logical_observable_id()
                            for t in inst.targets_copy())
            prob = float(inst.args_copy()[0])
            errors.append((dets, flips_log, prob))

    n_det = dem.num_detectors
    n_err = len(errors)
    pcm     = np.zeros((n_det, n_err), dtype=np.uint8)
    log_obs = np.zeros(n_err, dtype=np.uint8)
    probs   = np.zeros(n_err, dtype=np.float64)

    for j, (dets, fl, p) in enumerate(errors):
        for d in dets:
            pcm[d, j] = 1
        if fl:
            log_obs[j] = 1
        probs[j] = p

    return pcm, log_obs, probs


def probs_to_llrs(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert error probabilities to log-likelihood ratios: LLR = log((1-p)/p)."""
    p = np.clip(probs, eps, 1.0 - eps)
    return np.log((1.0 - p) / p)


# ---------------------------------------------------------------------------
# Detector → (ancilla, round) mapping
# ---------------------------------------------------------------------------

def build_detector_map(c_flat, rounds: int) -> Dict[int, Tuple[int, int]]:
    """
    Map detector index → (ancilla_qubit_index, round_index).

    Only single-ancilla detectors are mapped (XOR of consecutive measurements
    of the same ancilla qubit).  Multi-qubit detectors (final-round data+ancilla
    combinations) are excluded.

    Parameters
    ----------
    c_flat : stim.Circuit   flattened circuit
    rounds : int            number of syndrome rounds

    Returns
    -------
    dict {det_idx: (ancilla_q, round_idx)}
    """
    from circuit_builder import build_rec_to_qubit, infer_ticks_per_round

    rec_to_q = build_rec_to_qubit(c_flat)
    tpr = infer_ticks_per_round(c_flat, rounds)
    total_meas = len(rec_to_q)

    # Compute cumulative measurement count and round index per instruction
    insts = list(c_flat)
    cum = [0]
    tick = 0
    seen_first = False
    inst_round = []

    for inst in insts:
        if inst.name == "TICK":
            if not seen_first:
                seen_first = True
            else:
                tick += 1
        n_meas = (
            sum(1 for t in inst.targets_copy() if t.is_qubit_target)
            if inst.name in ("MR", "M", "MZ", "MX", "MY")
            else 0
        )
        cum.append(cum[-1] + n_meas)
        inst_round.append(tick // tpr if seen_first else 0)

    det_map: Dict[int, Tuple[int, int]] = {}
    det_idx = 0

    for i, inst in enumerate(insts):
        if inst.name == "DETECTOR":
            mb = cum[i + 1]
            ancs = list({
                rec_to_q[mb + t.value]
                for t in inst.targets_copy()
                if t.is_measurement_record_target and 0 <= mb + t.value < total_meas
            })
            if len(ancs) == 1:
                det_map[det_idx] = (ancs[0], inst_round[i])
            det_idx += 1

    return det_map


# ---------------------------------------------------------------------------
# Erasure detector and mechanism identification
# ---------------------------------------------------------------------------

def erased_detector_set(
    det_map: Dict[int, Tuple[int, int]],
    loss_events: List[Tuple[int, int]],
    reload_round,   # int, dict {qubit: round}, or None
    coords: dict,
    rounds: int,
) -> Set[int]:
    """
    Return the set of detector indices that are "erased" due to loss.

    A detector (ancilla a, round r) is erased if:
      - ancilla a is adjacent to qubit q, AND
      - r >= loss_round_of_q, AND
      - r <= reload_round_of_q (or end of circuit if no reload)

    Parameters
    ----------
    det_map       : {det_idx: (ancilla, round)} from build_detector_map
    loss_events   : list of (round, qubit) loss events
    reload_round  : int   — single reload round for all qubits, or
                    dict  — per-qubit {qubit: reload_round}, or
                    None  — no reload (erase until end of circuit)
    coords        : qubit coordinate dict
    rounds        : total rounds (used as upper bound when no reload)

    Returns
    -------
    set of erased detector indices
    """
    from circuit_builder import neighbor_ancillas_of_data

    # Build per-qubit loss round and adjacent ancilla sets
    qubit_loss_round: Dict[int, int] = {}
    qubit_adj_ancs: Dict[int, Set[int]] = {}
    for r, q in loss_events:
        q_int = int(q)
        r_int = int(r)
        if q_int not in qubit_loss_round or r_int < qubit_loss_round[q_int]:
            qubit_loss_round[q_int] = r_int
        if q_int not in qubit_adj_ancs:
            qubit_adj_ancs[q_int] = set(neighbor_ancillas_of_data(coords, q_int))

    def _end(q_int: int) -> int:
        if isinstance(reload_round, dict):
            rr = reload_round.get(q_int)
            return int(rr) if rr is not None else rounds
        elif reload_round is not None:
            return int(reload_round)
        return rounds

    erased: Set[int] = set()
    for det_idx, (anc, rnd) in det_map.items():
        for q_int, r_loss in qubit_loss_round.items():
            if anc in qubit_adj_ancs[q_int] and r_loss <= rnd <= _end(q_int):
                erased.add(det_idx)
                break
    return erased


def erasure_llrs(
    pcm: np.ndarray,
    probs: np.ndarray,
    erased_dets: Set[int],
    mode: str = "any",
) -> np.ndarray:
    """
    Return LLR vector with LLR = 0 for erasure-related mechanisms.

    For non-erased mechanisms: LLR = log((1-p)/p).
    For erasure mechanisms  : LLR = 0  (p_eff = 0.5, maximum uncertainty).

    Parameters
    ----------
    pcm        : (n_det, n_err) PCM
    probs      : (n_err,) error probabilities from DEM
    erased_dets: set of erased detector indices
    mode       : "any"  — mark mechanism as erasure if it fires ANY erased detector
                 "all"  — mark only if ALL its detectors are erased (stricter)

    Returns
    -------
    llrs : float64 array (n_err,)
    """
    llrs = probs_to_llrs(probs)

    if not erased_dets:
        return llrs

    erase_arr = sorted(erased_dets)
    fires_erased = pcm[erase_arr, :].any(axis=0)  # True where mech fires ≥1 erased det

    if mode == "any":
        erased_mech_mask = fires_erased
    else:  # "all": every detector the mechanism fires must be erased
        fires_any_det = pcm.any(axis=0)           # True where mech fires at least 1 det
        non_erased_arr = [i for i in range(pcm.shape[0]) if i not in erased_dets]
        fires_non_erased = (
            pcm[non_erased_arr, :].any(axis=0) if non_erased_arr
            else np.zeros(pcm.shape[1], dtype=bool)
        )
        # Erasure mechanism: fires at least one det, and all fired dets are erased
        erased_mech_mask = fires_any_det & fires_erased & ~fires_non_erased

    llrs[erased_mech_mask] = 0.0
    return llrs


# ---------------------------------------------------------------------------
# Union-Find decoder wrapper
# ---------------------------------------------------------------------------

class UFDecoder:
    """
    Thin wrapper around ldpc.UnionFindDecoder for the surface code.

    Parameters
    ----------
    pcm     : (n_det, n_err) binary PCM from dem_to_pcm
    log_obs : (n_err,) logical observable vector from dem_to_pcm
    """

    def __init__(self, pcm: np.ndarray, log_obs: np.ndarray):
        from ldpc.union_find_decoder import UnionFindDecoder
        # 'matrix' mode handles non-planar codes (DEM PCM can be non-planar)
        self._ufd = UnionFindDecoder(pcm, "matrix")
        self._log_obs = log_obs

    def decode_batch(
        self,
        det_events: np.ndarray,
        llrs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode a batch of syndrome shots.

        Parameters
        ----------
        det_events : uint8 array (trials, n_det)
        llrs       : float64 array (n_err,), or None for uniform LLR

        Returns
        -------
        logical_failures : bool array (trials,)
        """
        trials = det_events.shape[0]
        failures = np.zeros(trials, dtype=bool)

        for i in range(trials):
            syndrome = det_events[i].astype(np.uint8)
            if llrs is not None:
                correction = self._ufd.decode(syndrome, llrs=llrs)
            else:
                correction = self._ufd.decode(syndrome)
            log_correction = int(np.dot(correction, self._log_obs)) % 2
            failures[i] = bool(log_correction)

        return failures


# ---------------------------------------------------------------------------
# High-level evaluation functions
# ---------------------------------------------------------------------------

def decode_ler(
    c_flat,
    c_sample,
    trials: int = 1000,
) -> Dict:
    """
    Phase 3 — evaluate LER using UF decoder without erasure information.

    Uses the original circuit's DEM as the decoding model and samples from
    c_sample (may be a loss/reload circuit).

    Parameters
    ----------
    c_flat   : original noisy circuit (no loss) — used to build DEM / PCM
    c_sample : circuit to sample from (loss, reload, etc.)
    trials   : number of shots

    Returns
    -------
    dict with keys: logical_failure_rate, syndrome_intensity
    """
    dem = c_flat.detector_error_model()
    pcm, log_obs, _ = dem_to_pcm(dem)
    decoder = UFDecoder(pcm, log_obs)

    shots = c_sample.compile_detector_sampler().sample(trials, append_observables=True)
    det_ev  = shots[:, :-1].astype(np.uint8)
    raw_obs = shots[:, -1].astype(np.uint8)

    failures_from_decode = decoder.decode_batch(det_ev)
    # Net logical failure = decoder correction XOR raw observable
    logical_failures = failures_from_decode ^ raw_obs.astype(bool)

    return {
        "logical_failure_rate": float(logical_failures.mean()),
        "syndrome_intensity":   float(det_ev.sum(axis=1).mean()),
    }


def decode_ler_erasure(
    c_flat,
    c_sample,
    loss_events: List[Tuple[int, int]],
    reload_round: Optional[int],
    coords: dict,
    rounds: int,
    trials: int = 1000,
    erasure_mode: str = "any",
) -> Dict:
    """
    Evaluate LER using UF decoder with erasure information.

    Error mechanisms that fire detectors adjacent to lost qubits (during the
    loss period) receive LLR = 0, making the decoder treat them as erasures.

    Call convention
    ---------------
    P0 (no reload):
        c_sample = loss circuit (apply_loss_events without reload_info)
        reload_round = None
        → erasure covers [loss_round, rounds)

    P1 (oracle reload):
        c_sample = reload circuit (apply_loss_events with reload_info)
        reload_round = actual reload round
        → erasure covers [loss_round, reload_round)
        After reload_round, syndrome is near-normal; LLR returns to base values.

    P3 / P4 (inferred):
        Pass inferred_loss_events and inferred_reload_round instead of true values.

    Parameters
    ----------
    c_flat       : original noisy circuit (no loss) — DEM source
    c_sample     : circuit to sample from (loss or reload circuit)
    loss_events  : list of (round, qubit) loss events used to define erased region
    reload_round : round at which reload occurs (None = erased until end)
    coords       : qubit coordinate dict from get_qubit_coords
    rounds       : total circuit rounds
    trials       : number of shots
    erasure_mode : "any" — mark mechanism erased if it fires ANY erased detector
                   "all" — mark only if ALL its detectors are erased

    Returns
    -------
    dict with keys: logical_failure_rate, syndrome_intensity, n_erased_dets,
                    n_erased_mechs
    """
    dem = c_flat.detector_error_model()
    pcm, log_obs, probs = dem_to_pcm(dem)

    # Build erasure set
    det_map    = build_detector_map(c_flat, rounds)
    erase_dets = erased_detector_set(det_map, loss_events, reload_round, coords, rounds)
    llrs       = erasure_llrs(pcm, probs, erase_dets, mode=erasure_mode)
    n_erased_mechs = int((llrs == 0.0).sum())

    decoder = UFDecoder(pcm, log_obs)

    shots   = c_sample.compile_detector_sampler().sample(trials, append_observables=True)
    det_ev  = shots[:, :-1].astype(np.uint8)
    raw_obs = shots[:, -1].astype(np.uint8)

    failures_from_decode = decoder.decode_batch(det_ev, llrs=llrs)
    logical_failures     = failures_from_decode ^ raw_obs.astype(bool)

    return {
        "logical_failure_rate": float(logical_failures.mean()),
        "syndrome_intensity":   float(det_ev.sum(axis=1).mean()),
        "n_erased_dets":        len(erase_dets),
        "n_erased_mechs":       n_erased_mechs,
    }


# ---------------------------------------------------------------------------
# Convenience: compare protocols with UF decoder
# ---------------------------------------------------------------------------

def compare_protocols_uf(
    c_flat,
    loss_events: List[Tuple[int, int]],
    reload_round: int,
    coords: dict,
    rounds: int,
    trials: int = 2000,
    erasure_mode: str = "any",
) -> Dict:
    """
    Compare P0 (no reload) vs P1 (oracle erasure reload) using UF decoder.

    P0: samples from loss circuit, uses base LLRs (no erasure info)
    P1: samples from reload circuit, uses LLR=0 for erased mechanisms

    Note on logical-string qubits
    ------------------------------
    Loss of qubits ON the logical X string causes LER ≈ 0.5 for BOTH P0 and P1
    under the destructive-reset model (Phase 3).  The reset qubit has no memory
    of the original logical state, so the final X measurement is random.
    Meaningful P0 vs P1 improvement is visible for qubits OFF the logical string,
    where oracle reload removes syndrome confusion without affecting the observable.

    Phase 4 (Union-Find + erasure) properly handles logical-string losses via the
    peeling decoder on the erasure cluster.

    Parameters
    ----------
    c_flat       : original noisy circuit (no loss)
    loss_events  : list of (round, qubit) loss events
    reload_round : oracle reload round (for P1)
    coords, rounds, trials : as in decode_ler
    erasure_mode : "any" or "all" — see erasure_llrs

    Returns
    -------
    dict with P0 and P1 metrics, delta_LER, erased detector/mechanism counts
    """
    from circuit_builder import apply_loss_events

    if not loss_events:
        r = decode_ler(c_flat, c_flat, trials)
        return {k: r["logical_failure_rate"] if "LER" in k else
                   r["syndrome_intensity"] if "SI" in k else 0
                for k in ("P0_LER", "P0_SI", "P1_LER", "P1_SI",
                          "delta_LER", "n_erased_dets", "n_erased_mechs")}

    c_p0 = apply_loss_events(c_flat, loss_events, rounds)
    c_p1 = apply_loss_events(
        c_flat, loss_events, rounds,
        reload_info={int(q): reload_round for _, q in loss_events},
    )

    r_p0 = decode_ler(c_flat, c_p0, trials)
    r_p1 = decode_ler_erasure(c_flat, c_p1, loss_events, reload_round,
                               coords, rounds, trials,
                               erasure_mode=erasure_mode)

    return {
        "P0_LER":         r_p0["logical_failure_rate"],
        "P0_SI":          r_p0["syndrome_intensity"],
        "P1_LER":         r_p1["logical_failure_rate"],
        "P1_SI":          r_p1["syndrome_intensity"],
        "delta_LER":      r_p0["logical_failure_rate"] - r_p1["logical_failure_rate"],
        "n_erased_dets":  r_p1["n_erased_dets"],
        "n_erased_mechs": r_p1["n_erased_mechs"],
    }


# Legacy decoders (PeelingErasureDecoder, SpaceTimePeelingDecoder,
# decode_ler_peeling, loss_to_erased_qubit_rounds) live in old_decoder.py.


