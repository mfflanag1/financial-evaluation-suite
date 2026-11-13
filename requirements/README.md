# Runtime dependency constraints

`runtime-constraints.txt` records the resolved third-party runtime versions used
for Python 3.12 native validation. Local financial packages are installed together
from this checkout and are not represented by personal machine paths.

The constraint file pins versions; it is not a hash-locked supply-chain manifest
or a promise of bit-identical wheels. Platform wheels and isolated build-system
dependencies are resolved separately. Keep the constraints unchanged when
comparing runtime versions. Upgrade them only with a fresh install and the native
and Docker checks.

`scripts/bootstrap.py` creates a new virtual environment and installs every local
package in one resolver invocation. It refuses to overwrite an existing path.
Offline use requires a wheelhouse containing both runtime and build dependencies.

Installation finishes with an isolated subprocess outside the checkout. It
compares all installed project resources with the source inventory, rejects
editable or checkout imports, and discovers the pinned native plugins. Run that
check independently with the installed interpreter and `scripts/check_installed.py`.
