"""Circuit construction and loss injection for tsim, at three noise levels.

The loss model (puncturing: CX gates touching a lost qubit are removed,
optionally dressed with a Pauli channel on the surviving qubit; optional
reload and LDU noise) is ported unchanged from ``src/circuit_builder.py`` —
see that module's docstring for the physical motivation. It is applied on a
flattened `stim.Circuit` and only *then* wrapped into a `tsim.Circuit`, since
tsim circuits are thin wrappers around stim circuits
(`tsim.Circuit.from_stim_program`).

What's new here relative to ``src/circuit_builder.py``:

- `noise_models.CodeCapacityNoise` / `PhenomenologicalNoise` / `CircuitLevelNoise`
  select which stim.Circuit.generated() noise kwargs get used; loss injection
  is orthogonal and works the same way at all three levels.
- `infer_ticks_per_round` gets a fallback for `rounds <= 1` (the code-capacity
  case), where there aren't enough repeated ancilla measurements to infer the
  round length by counting — the whole circuit *is* round 0.
- `detector_error_model` / `sample_detectors` wrap tsim's non-deterministic-
  detector-aware sampler, which is the reason to use tsim here at all: loss
  puncturing routinely produces non-deterministic detectors (punctured
  stabilizer generators can anticommute), which stim's own
  `detector_error_model()` refuses to analyze.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import stim
import tsim

from tlsim.noise_models import CircuitLevelNoise, CodeCapacityNoise, NoiseModel, PhenomenologicalNoise

# ---------------------------------------------------------------------------
# Coordinate / topology helpers (unchanged from src/circuit_builder.py)
# ---------------------------------------------------------------------------


def get_qubit_coords(c: stim.Circuit) -> dict:
    """Return {qubit_index: (x, y, ...)} from QUBIT_COORDS instructions."""
    coords = {}
    for inst in c:
        if inst.name == "QUBIT_COORDS":
            xy = tuple(inst.gate_args_copy())
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    coords[t.value] = xy
    return coords


def find_qubit_by_coord(coords: dict, xy: Tuple) -> int:
    """Return qubit index whose coordinate matches xy (first two components)."""
    for q, cxy in coords.items():
        if len(cxy) >= 2 and (cxy[0], cxy[1]) == xy:
            return q
    raise ValueError(f"No qubit with coord={xy}")


def data_qubits(coords: dict) -> List[int]:
    """Return data qubit indices (odd, odd) coords in rotated surface code."""
    return [
        q for q, xy in coords.items()
        if len(xy) >= 2
        and int(round(xy[0])) % 2 == 1
        and int(round(xy[1])) % 2 == 1
    ]


def ancilla_qubits(coords: dict) -> List[int]:
    """Return ancilla qubit indices (not odd-odd) in rotated surface code."""
    ds = set(data_qubits(coords))
    return [q for q in coords if q not in ds]


def neighbor_ancillas_of_data(coords: dict, data_q: int) -> List[int]:
    """Return ancilla qubits adjacent (diagonally) to data_q."""
    x, y = coords[data_q][:2]
    want = {(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)}
    return sorted(
        [q for q, xy in coords.items() if len(xy) >= 2 and (xy[0], xy[1]) in want],
        key=lambda q: coords[q],
    )


def neighbor_data_of_ancilla(coords: dict, ancilla_q: int) -> List[int]:
    """Return data qubits adjacent (diagonally) to ancilla_q."""
    x, y = coords[ancilla_q][:2]
    want = {(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)}
    ds = set(data_qubits(coords))
    return sorted(
        [q for q, xy in coords.items() if len(xy) >= 2 and (xy[0], xy[1]) in want and q in ds],
        key=lambda q: coords[q],
    )


def choose_logical_x_string(coords: dict) -> List[int]:
    """Return data qubits on the leftmost column — a simple logical-X representative."""
    ds = data_qubits(coords)
    x0 = sorted({coords[q][0] for q in ds})[0]
    return sorted([q for q in ds if coords[q][0] == x0], key=lambda q: coords[q][1])


# ---------------------------------------------------------------------------
# Measurement record helpers (unchanged from src/circuit_builder.py)
# ---------------------------------------------------------------------------


def build_rec_to_qubit(c: stim.Circuit) -> List[int]:
    """Return list mapping measurement index -> qubit index."""
    rec_to_qubit = []
    for inst in c:
        if inst.name in ("MR", "M", "MZ", "MX", "MY"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    rec_to_qubit.append(t.value)
    return rec_to_qubit


def qubit_to_rec_series(rec_to_qubit: List[int]) -> dict:
    """Return {qubit: [rec_index, ...]} (all measurement indices for each qubit)."""
    m = defaultdict(list)
    for k, q in enumerate(rec_to_qubit):
        m[q].append(k)
    return dict(m)


def build_rec_to_qubit_and_inst_index(
    c: stim.Circuit,
) -> Tuple[List[int], List[int]]:
    """Return (rec_to_qubit, rec_to_inst_index) parallel lists."""
    rec_to_qubit, rec_to_inst_i = [], []
    for i, inst in enumerate(c):
        if inst.name in ("MR", "M", "MZ", "MX", "MY"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    rec_to_qubit.append(t.value)
                    rec_to_inst_i.append(i)
    return rec_to_qubit, rec_to_inst_i


# ---------------------------------------------------------------------------
# Round timing inference
# ---------------------------------------------------------------------------


def infer_ticks_per_round(c_flat: stim.Circuit, rounds: int) -> int:
    """Infer the number of TICK instructions per syndrome round.

    Uses the most-measured ancilla qubit's consecutive measurement intervals
    rather than total-tick / rounds, which can be off due to tail handling.

    Falls back to "the whole circuit is one round" when `rounds <= 1` (the
    code-capacity case): there, an ancilla is measured only once via MR, so
    there aren't enough repeated measurements to infer an interval from.
    """
    tick_total = sum(1 for inst in c_flat if inst.name == "TICK")
    if rounds <= 1:
        return max(tick_total, 1)

    rec_to_q, rec_to_inst_i = build_rec_to_qubit_and_inst_index(c_flat)
    q_to_recs = defaultdict(list)
    for k, q in enumerate(rec_to_q):
        q_to_recs[q].append(k)

    q_pick = max(q_to_recs.keys(), key=lambda q: len(q_to_recs[q]))
    rec_ids = q_to_recs[q_pick]
    if len(rec_ids) < 3:
        raise RuntimeError("Not enough repeated measurements to infer ticks_per_round.")

    c_list = list(c_flat)
    tick_prefix = np.zeros(len(c_list) + 1, dtype=int)
    t = 0
    for i, inst in enumerate(c_list):
        if inst.name == "TICK":
            t += 1
        tick_prefix[i + 1] = t

    kmax = min(len(rec_ids) - 1, rounds, 20)
    diffs = [
        int(tick_prefix[rec_to_inst_i[rec_ids[k + 1]] + 1]
            - tick_prefix[rec_to_inst_i[rec_ids[k]] + 1])
        for k in range(kmax)
    ]
    vals, counts = np.unique(diffs, return_counts=True)
    tpr = int(vals[np.argmax(counts)])
    if tpr <= 0:
        raise RuntimeError(f"Invalid ticks_per_round inferred: {tpr}")
    return tpr


# ---------------------------------------------------------------------------
# Loss injection — puncturing model (ported from src/circuit_builder.py)
# ---------------------------------------------------------------------------

_SURVIVOR_NOISE_GATE = {
    "depolarize": "DEPOLARIZE1",
    "X": "X_ERROR",
    "Y": "Y_ERROR",
    "Z": "Z_ERROR",
}

_NOISE_1Q = frozenset({"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1"})
_NOISE_2Q = frozenset({"DEPOLARIZE2", "PAULI_CHANNEL_2"})


def apply_loss_events(
    c_flat: stim.Circuit,
    loss_events: List[Tuple[int, int]],
    rounds: int,
    *,
    p_surv: float = 0.0,
    survivor_mode: str = "depolarize",
    reload_info: Optional[dict] = None,
    reload_basis: str = "X",
    p_reload: float = 0.0,
    p_ldu_propagation: float = 0.0,
    ldu_round: Optional[int] = None,
    ldu_target_qubits: Optional[List[int]] = None,
    p_ldu_noise: float = 0.0,
) -> stim.Circuit:
    """Inject loss events into a flattened Stim circuit.

    Parameters
    ----------
    c_flat : stim.Circuit
        Flattened circuit (no REPEAT blocks).
    loss_events : list of (round, qubit) pairs
        Each entry specifies the round at which qubit *qubit* is lost.
        CX gates involving that qubit are removed from *round* onward.
    rounds : int
        Total number of syndrome rounds (used to infer ticks_per_round).
    p_surv : float
        Probability of Pauli channel applied to the survivor qubit of each
        removed CX gate (Model B). Set to 0 for Model A (no survivor noise).
    survivor_mode : str
        One of "none", "depolarize", "X", "Y", "Z". Ignored when p_surv == 0.
    reload_info : dict {qubit: reload_round}, optional
        Specifies when each lost qubit is reloaded (reset and returned to the
        circuit). At reload_round, an RX/RZ gate is inserted and the qubit is
        removed from active_lost so CX gates are restored for all subsequent
        rounds.
    reload_basis : str
        "X" or "Z" reset basis for the reload gate.
    p_reload : float
        Depolarizing noise applied to the qubit immediately after reload
        (physical reset noise).
    p_ldu_propagation : float
        Additional DEPOLARIZE1 applied to the RELOADED qubit at the reload
        round, modelling back-propagation noise from the LDU circuit at the
        reload site. Applied BEFORE p_reload.
    ldu_round : int, optional
        Round at which the LDU circuit is executed on data qubits (models
        circuit-based LDU, arXiv:2405.10434): DEPOLARIZE1(p_ldu_noise) is
        applied to every PRESENT (non-lost) target qubit at this round.
    ldu_target_qubits : list of int, optional
        Data qubits on which the LDU circuit runs. None = no LDU noise.
    p_ldu_noise : float
        Per-qubit DEPOLARIZE1 noise from running the LDU circuit, applied to
        each PRESENT qubit in ldu_target_qubits at ldu_round.

    Returns
    -------
    stim.Circuit
        Modified circuit with loss events (and optional reloads) applied.
    """
    if not loss_events:
        return c_flat

    tpr = infer_ticks_per_round(c_flat, rounds)

    by_round = defaultdict(list)
    for r, q in loss_events:
        r, q = int(r), int(q)
        if 0 <= r < max(rounds, 1):
            by_round[r].append(q)

    _reload_info: dict = {}
    if reload_info:
        for q, rr in reload_info.items():
            _reload_info[int(q)] = int(rr)

    add_survivor_noise = p_surv > 0 and survivor_mode != "none"
    if add_survivor_noise and survivor_mode not in _SURVIVOR_NOISE_GATE:
        raise ValueError(
            f"survivor_mode must be one of {list(_SURVIVOR_NOISE_GATE)}, got '{survivor_mode}'"
        )
    noise_gate = _SURVIVOR_NOISE_GATE.get(survivor_mode, "DEPOLARIZE1")
    reset_gate = "RX" if reload_basis.upper() == "X" else "RZ"

    out = stim.Circuit()
    seen_first_tick = False
    tick_after_init = 0
    last_round = None
    active_lost: set = set()
    reloaded_this_round: set = set()

    for inst in c_flat:
        if inst.name == "TICK":
            out.append(inst)
            if not seen_first_tick:
                seen_first_tick = True
            else:
                tick_after_init += 1
            reloaded_this_round = set()
            continue

        if not seen_first_tick:
            out.append(inst)
            continue

        current_round = tick_after_init // tpr

        if current_round != last_round:
            active_lost.update(by_round.get(current_round, []))

            if (ldu_round is not None
                    and current_round == ldu_round
                    and ldu_target_qubits is not None
                    and p_ldu_noise > 0):
                present_targets = sorted(
                    q for q in ldu_target_qubits if q not in active_lost
                )
                if present_targets:
                    out.append("DEPOLARIZE1", present_targets, p_ldu_noise)

            if _reload_info:
                to_reload = [q for q, rr in _reload_info.items()
                             if rr == current_round and q in active_lost]
                if to_reload:
                    active_lost -= set(to_reload)
                    reloaded_this_round = set(to_reload)
                    if p_ldu_propagation > 0:
                        out.append("DEPOLARIZE1", sorted(to_reload), p_ldu_propagation)
                    out.append(reset_gate, sorted(to_reload))
                    if p_reload > 0:
                        out.append("DEPOLARIZE1", sorted(to_reload), p_reload)

            last_round = current_round

        if inst.name == "CX" and active_lost:
            tg = inst.targets_copy()
            kept_pairs: List[Tuple[int, int]] = []
            survivors: List[int] = []

            for a, b in zip(tg[0::2], tg[1::2]):
                qa, qb = a.value, b.value
                if qa in active_lost:
                    survivors.append(qb)
                elif qb in active_lost:
                    survivors.append(qa)
                else:
                    kept_pairs.append((qa, qb))

            if kept_pairs:
                out.append("CX", [q for pair in kept_pairs for q in pair])
            if add_survivor_noise and survivors:
                out.append(noise_gate, survivors, p_surv)
            continue

        if inst.name in _NOISE_1Q and active_lost:
            kept = [t.value for t in inst.targets_copy() if t.value not in active_lost]
            if kept:
                args = inst.gate_args_copy()
                out.append(inst.name, kept, args[0] if args else 0)
            continue

        if inst.name in _NOISE_2Q and active_lost:
            tg = inst.targets_copy()
            args = inst.gate_args_copy()
            kept = []
            for a, b in zip(tg[0::2], tg[1::2]):
                if a.value not in active_lost and b.value not in active_lost:
                    kept.extend([a.value, b.value])
            if kept:
                out.append(inst.name, kept, args[0] if len(args) == 1 else args)
            continue

        out.append(inst)

    return out


def apply_single_loss(
    c_flat: stim.Circuit,
    q_erase: int,
    loss_round: int,
    rounds: int,
    *,
    p_surv: float = 0.0,
    survivor_mode: str = "depolarize",
) -> stim.Circuit:
    """Convenience wrapper for a single loss event."""
    return apply_loss_events(
        c_flat,
        [(loss_round, q_erase)],
        rounds,
        p_surv=p_surv,
        survivor_mode=survivor_mode,
    )


def sample_loss_events(
    data_qubit_list: List[int],
    rounds: int,
    p_loss: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Sample loss events under a per-round independent Bernoulli model.

    For each data qubit, independently sample the first round at which it is
    lost; once lost, it remains lost for the rest of the circuit (no reload).

    Returns list of (round, qubit) pairs, sorted by round.
    """
    if p_loss <= 0.0:
        return []
    rounds = max(rounds, 1)
    events = []
    for q in data_qubit_list:
        rolls = rng.random(rounds)
        first_loss = np.argmax(rolls < p_loss)
        if rolls[first_loss] < p_loss:
            events.append((int(first_loss), int(q)))
    return sorted(events)


# ---------------------------------------------------------------------------
# Noise rescaling helper (unchanged from src/circuit_builder.py)
# ---------------------------------------------------------------------------


def _rescale_depolarize1(circuit: stim.Circuit, new_p: float) -> stim.Circuit:
    """Replace all DEPOLARIZE1 probabilities in a flattened circuit with new_p."""
    s = str(circuit)
    if new_p > 0:
        s = re.sub(r'DEPOLARIZE1\([^)]+\)', f'DEPOLARIZE1({new_p})', s)
    else:
        s = '\n'.join(l for l in s.split('\n')
                      if not l.strip().startswith('DEPOLARIZE1'))
    return stim.Circuit(s)


# ---------------------------------------------------------------------------
# Noise-level-aware circuit builder
# ---------------------------------------------------------------------------


def build_surface_code_circuit(
    d: int,
    rounds: int,
    noise: NoiseModel,
    memory_basis: str = "X",
) -> Tuple[stim.Circuit, stim.Circuit]:
    """Return (original_circuit, flattened_circuit) for a rotated surface code
    memory experiment at the ambient noise level given by `noise`.

    Parameters
    ----------
    d : int
        Code distance.
    rounds : int
        Number of syndrome rounds. Must be 1 for `CodeCapacityNoise`.
    noise : CodeCapacityNoise | PhenomenologicalNoise | CircuitLevelNoise
        Selects which stim.Circuit.generated() noise kwargs are used.
    memory_basis : str
        "X" or "Z" memory experiment.

    Returns
    -------
    (c0, c_flat) : (stim.Circuit, stim.Circuit)
    """
    kwargs: dict = dict(distance=d, rounds=rounds)
    rescale_p_1Q: Optional[float] = None
    rescale_base_p: float = 0.0

    if isinstance(noise, CodeCapacityNoise):
        if rounds != 1:
            raise ValueError(
                f"CodeCapacityNoise requires rounds=1 (single idealized "
                f"syndrome-extraction round), got rounds={rounds}."
            )
        if noise.p_data > 0:
            kwargs["before_round_data_depolarization"] = noise.p_data
    elif isinstance(noise, PhenomenologicalNoise):
        if noise.p_data > 0:
            kwargs["before_round_data_depolarization"] = noise.p_data
        if noise.p_meas > 0:
            kwargs["before_measure_flip_probability"] = noise.p_meas
    elif isinstance(noise, CircuitLevelNoise):
        if noise.p_pauli > 0:
            kwargs["after_clifford_depolarization"] = noise.p_pauli
        if noise.p_meas > 0:
            kwargs["before_measure_flip_probability"] = noise.p_meas
        rescale_p_1Q = noise.p_1Q
        rescale_base_p = noise.p_pauli
    else:
        raise TypeError(f"Unrecognized noise model: {noise!r}")

    c0 = stim.Circuit.generated(
        f"surface_code:rotated_memory_{memory_basis.lower()}",
        **kwargs,
    )
    c_flat = c0.flattened()
    if rescale_p_1Q is not None and rescale_base_p > 0 and rescale_p_1Q != rescale_base_p:
        c_flat = _rescale_depolarize1(c_flat, rescale_p_1Q)
    return c0, c_flat


# ---------------------------------------------------------------------------
# tsim wrapping / sampling
# ---------------------------------------------------------------------------


def to_tsim(circuit: stim.Circuit) -> "tsim.Circuit":
    """Wrap a (possibly loss-punctured) stim.Circuit as a tsim.Circuit."""
    return tsim.Circuit.from_stim_program(circuit)


def build_lossy_tsim_circuit(
    d: int,
    rounds: int,
    noise: NoiseModel,
    loss_events: List[Tuple[int, int]],
    *,
    memory_basis: str = "X",
    p_surv: float = 0.0,
    survivor_mode: str = "depolarize",
    reload_info: Optional[dict] = None,
    reload_basis: str = "X",
    p_reload: float = 0.0,
    p_ldu_propagation: float = 0.0,
    ldu_round: Optional[int] = None,
    ldu_target_qubits: Optional[List[int]] = None,
    p_ldu_noise: float = 0.0,
) -> "tsim.Circuit":
    """Build a rotated surface code circuit at the given noise level, inject
    loss events (puncturing model), and return it wrapped as a tsim.Circuit.

    This is the main entry point: combines `build_surface_code_circuit` +
    `apply_loss_events` + `to_tsim`.
    """
    _, c_flat = build_surface_code_circuit(d, rounds, noise, memory_basis=memory_basis)
    c_loss = apply_loss_events(
        c_flat,
        loss_events,
        rounds,
        p_surv=p_surv,
        survivor_mode=survivor_mode,
        reload_info=reload_info,
        reload_basis=reload_basis,
        p_reload=p_reload,
        p_ldu_propagation=p_ldu_propagation,
        ldu_round=ldu_round,
        ldu_target_qubits=ldu_target_qubits,
        p_ldu_noise=p_ldu_noise,
    )
    return to_tsim(c_loss)


def detector_error_model(
    circuit: "tsim.Circuit",
    *,
    allow_gauge_detectors: bool = True,
    **kwargs,
) -> stim.DetectorErrorModel:
    """Return the detector error model of a (possibly lossy) tsim circuit.

    Defaults `allow_gauge_detectors=True` since loss puncturing routinely
    produces non-deterministic detectors/observables (punctured stabilizer
    generators can anticommute) — this is exactly the case stim's own
    `detector_error_model()` refuses to analyze, and the reason to prefer
    tsim for loss circuits in the first place.
    """
    return circuit.detector_error_model(allow_gauge_detectors=allow_gauge_detectors, **kwargs)


def sample_detectors(
    circuit: "tsim.Circuit",
    shots: int,
    *,
    stim_compatible: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample detector/observable outcomes from a (possibly lossy) tsim circuit.

    tsim reports *actual* detector values by default (`stim_compatible=False`,
    the default here), whereas stim reports detection *flips* (samples XORed
    against a noiseless reference sample). Setting `stim_compatible=True` XORs
    against a noiseless reference sample to match stim's convention — but per
    tsim's own docs this should only be used when detectors/observables are
    deterministic. Loss-punctured circuits routinely are *not* (that's the
    whole reason to prefer tsim's DEM here, see `detector_error_model`), so
    the default is the always-well-defined raw/absolute sample. Only pass
    `stim_compatible=True` once you've confirmed the circuit's detectors are
    deterministic (e.g. no loss, or a noise level where puncturing doesn't
    break commutativity).
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(
        shots=shots,
        use_detector_reference_sample=stim_compatible,
        use_observable_reference_sample=stim_compatible,
    )
