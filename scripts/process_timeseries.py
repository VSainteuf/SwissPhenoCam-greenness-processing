#!/usr/bin/env python
"""Stage 2 script stub — delegates to swissphenocam.cli.process.

The driver logic lives in ``src/swissphenocam/cli/process.py`` so it can be
installed as a console script (``swissphenocam-process``).
"""

# Stage 2

from swissphenocam.cli.process import main

if __name__ == "__main__":
    main()
