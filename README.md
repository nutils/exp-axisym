# exp-axisym
Experimental tools for axial symmetry

The `nutils.experimental` modules contain constructs for which no canonical
form has yet been found, and which may be of use in the interim. They are to be
installed directly from Github with reference to the specific commit, as
backwards compatibility is in no way guaranteed.

This is the structure of the required pyproject.toml entry:

```toml
[project]
...
dependencies = [
    ...
    "nutils-experimental-axisym @ git+https://github.com/nutils/exp-axisym.git@[commit-hash]"
]
