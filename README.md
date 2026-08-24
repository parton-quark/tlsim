# loss_tolerance

Loss-error simulation for quantum error-correcting circuits, built on top of
[tsim](https://github.com/QuEraComputing/tsim) (QuEra's GPU-accelerated,
ZX-calculus-based circuit sampler, with an API that closely mirrors
[stim](https://github.com/quantumlib/Stim)).

## Structure

```
src/
├── tlsim/                    # the package — everything needed to build and decode lossy circuits
│   ├── __init__.py           # public API re-exports
│   ├── noise_models.py       # CodeCapacityNoise / PhenomenologicalNoise / CircuitLevelNoise
│   ├── circuit_builder.py    # surface-code construction + loss puncturing + tsim DEM/sampling
│   ├── decode.py             # LDUMode + decode_ler_with_ldu (LER via the union_find UF decoder)
│   ├── test_tlsim.py         # pytest sanity checks
│   ├── usage_example.ipynb   # walkthrough notebook (Japanese)
│   └── usage_example_en.ipynb  # same walkthrough, English
└── union_find.py             # erasure-aware Union-Find decoder (ldpc.UnionFindDecoder wrapper),
                               # used by tlsim.decode
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install bloqade-tsim stim numpy scipy matplotlib pytest ldpc
```

`tsim` is distributed as the `bloqade-tsim` package.

## Usage

```python
import sys
sys.path.insert(0, "src")

from tlsim.circuit_builder import build_surface_code_circuit, build_lossy_tsim_circuit, sample_detectors
from tlsim.noise_models import CircuitLevelNoise

noise = CircuitLevelNoise(p_pauli=0.005, p_meas=0.005)
d, rounds = 5, 9
q_lost = 12  # a data-qubit index from build_surface_code_circuit's coords

tc = build_lossy_tsim_circuit(d, rounds, noise, loss_events=[(2, q_lost)])
samples = sample_detectors(tc, shots=2000)
```

To also get a logical error rate, use `tlsim.decode`:

```python
from tlsim.decode import LDUMode, decode_ler_with_ldu

mode = LDUMode(kind="noisy", ldu_latency=1, reload_latency=1, p_ldu_gate=0.03, p_reload=0.01)
result = decode_ler_with_ldu(c_flat, loss_events=[(2, q_lost)], rounds=rounds, mode=mode, trials=3000, seed=1)
print(result["logical_failure_rate"])
```

See `usage_example_en.ipynb` for a full walkthrough — all three noise levels,
the non-deterministic-detector case that motivates using tsim, an
LDU/reload comparison, and LER sweeps over code distance and physical error
rate.

## Tests

```bash
pytest src/tlsim/test_tlsim.py -v
```

## Notes

- `tlsim.decode` computes the detector error model directly from the actual
  loss/reload circuit (`detector_error_model(..., allow_gauge_detectors=True)`),
  rather than approximating it from the loss-free circuit's DEM with
  erasure LLRs patched in. The approximation was found to make
  `ldpc.UnionFindDecoder` hang on syndromes its (mismatched) PCM can't
  explain whenever puncturing produces gauge detectors — see the module
  docstring in `src/tlsim/decode.py` for details.
- `src/union_find.py` wraps `ldpc.union_find_decoder.UnionFindDecoder`
  (Delfosse–Nickerson Union-Find, generalized to general hypergraph PCMs)
