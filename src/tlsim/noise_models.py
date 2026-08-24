"""Ambient (non-loss) noise level definitions.

Loss/erasure is handled separately (see ``circuit_builder.apply_loss_events``)
since it applies identically regardless of which ambient noise level below is
in effect — puncturing removes CX gates touching the lost qubit from whatever
circuit these noise levels produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class CodeCapacityNoise:
    """Single idealized syndrome-extraction round.

    Data qubits are depolarized once (probability `p_data`) before a single,
    otherwise noiseless, round of stabilizer measurement. There is no gate
    noise and no measurement noise — this is the standard "code capacity"
    abstraction. Must be paired with ``rounds=1``.
    """

    p_data: float = 0.0


@dataclass(frozen=True)
class PhenomenologicalNoise:
    """Noisy data qubits + noisy measurement outcomes, perfect gates.

    `p_data` is applied as `DEPOLARIZE1(p_data)` on every data qubit at the
    start of each round (stim's `before_round_data_depolarization`). `p_meas`
    is a bit-flip probability applied immediately before each measurement
    (stim's `before_measure_flip_probability`). No `after_clifford_depolarization`.
    """

    p_data: float = 0.0
    p_meas: float = 0.0


@dataclass(frozen=True)
class CircuitLevelNoise:
    """Full gate-level noise, matching ``src/circuit_builder.py``.

    `p_pauli` is `after_clifford_depolarization` (treated as the 2Q gate error
    rate). `p_meas` is `before_measure_flip_probability`. `p_1Q`, if given,
    rescales the DEPOLARIZE1 (1Q gate) probability independently of `p_pauli`
    (typical: `p_1Q = p_pauli / 10`), leaving DEPOLARIZE2 (2Q gates) at `p_pauli`.
    """

    p_pauli: float = 0.0
    p_meas: float = 0.0
    p_1Q: float | None = None


NoiseModel = Union[CodeCapacityNoise, PhenomenologicalNoise, CircuitLevelNoise]
