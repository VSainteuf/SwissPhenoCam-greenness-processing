#!/usr/bin/env python
"""Stage 1 script stub — delegates to swissphenocam.cli.extract.

The driver logic lives in ``src/swissphenocam/cli/extract.py`` so it can be
installed as a console script (``swissphenocam-extract``).
"""

# Stage 1

from swissphenocam.cli.extract import main

if __name__ == "__main__":
    main()
