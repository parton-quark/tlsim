"""tlsim: loss-error support for tsim (https://github.com/QuEraComputing/tsim).

Ports the puncturing-based loss model developed in ``src/circuit_builder.py``
(qubit loss = removal of CX gates touching the lost qubit, i.e. the stabilizer
generators are "punctured", optionally dressed with a Pauli channel on the
surviving qubit of each removed CX) so it can drive circuits built for tsim
instead of plain stim.

Three ambient noise levels are supported, selected via `noise_models`:

- `CodeCapacityNoise`  — single idealized syndrome-extraction round, noise only
  on data qubits before that round.
- `PhenomenologicalNoise` — perfect gates, but noisy data qubits between rounds
  and noisy measurement outcomes (``before_round_data_depolarization`` /
  ``before_measure_flip_probability``).
- `CircuitLevelNoise` — full gate-level noise (``after_clifford_depolarization``
  / ``before_measure_flip_probability``), matching ``src/circuit_builder.py``.

Loss injection (puncturing, survivor noise, reload, LDU noise) is orthogonal to
the ambient noise level and is applied via `circuit_builder.apply_loss_events`
on the flattened stim circuit before wrapping it as a `tsim.Circuit`.

The main reason to prefer tsim here: circuits with loss punctures often have
non-deterministic detectors/observables (puncturing can make previously
commuting stabilizer generators anticommute). Stim's own
``detector_error_model()`` refuses such circuits, which is why the rest of
this project had to fall back on a custom Union-Find / peeling decoder. tsim's
analytic ZX sampler supports non-deterministic detectors natively, so
``Circuit.detector_error_model(allow_gauge_detectors=True)`` succeeds where
stim fails (see `circuit_builder.detector_error_model`).
"""

from tlsim.circuit_builder import (
    apply_loss_events,
    apply_single_loss,
    build_lossy_tsim_circuit,
    build_surface_code_circuit,
    detector_error_model,
    sample_detectors,
    sample_loss_events,
    to_tsim,
)
from tlsim.decode import LDUMode, build_ldu_circuit, decode_ler_with_ldu
from tlsim.noise_models import (
    CircuitLevelNoise,
    CodeCapacityNoise,
    NoiseModel,
    PhenomenologicalNoise,
)

__all__ = [
    "CircuitLevelNoise",
    "CodeCapacityNoise",
    "LDUMode",
    "NoiseModel",
    "PhenomenologicalNoise",
    "apply_loss_events",
    "apply_single_loss",
    "build_ldu_circuit",
    "build_lossy_tsim_circuit",
    "build_surface_code_circuit",
    "decode_ler_with_ldu",
    "detector_error_model",
    "sample_detectors",
    "sample_loss_events",
    "to_tsim",
]
