This is a separate Python 3.12 CPU installation. The frozen evaluation environment
requires Verifiers 0.4.1 and must not be reused or upgraded in place.

`sources.json` binds the four native source repositories to exact commits and real
distribution versions. The bootstrap requires clean existing checkouts, including
their Git tags for VCS versioning. It builds ordinary wheels; it does not override
metadata, omit dependencies, install editable packages, or add source overlays.
The financial core packages are installed without their evaluation-only extras.

`cpu-requirements.txt` is a hash-locked registry dependency closure for the native
base packages. It was resolved for Python 3.12 on macOS arm64 using compatible pins
from the pinned trainer's lock. It is a separate CPU lock, not the upstream GPU
lock. In particular, LiteLLM 1.106.2 supplies a compatible binary wheel where the
upstream 1.95.0 pin required a newer Rust build toolchain. No GPU extras or torch
are installed. Other platforms must pass the same installation and import checks
before being claimed supported. This installation still includes upstream base
dependencies such as Harbor and monitoring clients; no monitor or provider is
started by the bootstrap.

Run from the repository root with an unused output directory outside the checkout:

```sh
python3 training/scripts/bootstrap.py \
  --python "$PYTHON312" --out "$TRAINING_INSTALL" \
  --prime-rl-source "$PRIME_RL_SOURCE" \
  --verifiers-source "$VERIFIERS_SOURCE" \
  --renderers-source "$RENDERERS_SOURCE" \
  --pydantic-config-source "$PYDANTIC_CONFIG_SOURCE"
```

Use `--offline` only when build requirements and registry wheels are already
cached. The bootstrap verifies source commits and gitlinks, builds pinned wheels,
installs with hash checking and full dependency resolution, runs `uv pip check`,
and checks isolated installed imports and wheel resource bytes. It records a
machine-readable report and logs outside the checkout. A successful result proves
CPU packaging and imports only; it does not establish a live rollout, GPU trainer,
advantage computation, or optimizer step. Native GPU installation and training
require a separate explicitly authorized gate.
