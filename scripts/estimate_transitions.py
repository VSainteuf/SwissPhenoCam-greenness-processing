#!/usr/bin/env python
"""Stage 3 script stub — delegates to swissphenocam.cli.transitions.

The driver logic lives in ``src/swissphenocam/cli/transitions.py`` so it can be
installed as a console script (``swissphenocam-transitions``).

Each run processes a single chromatic index (GCC, RCC, or BCC) as set by
``series_index`` in the config.  To estimate transitions for all three indices,
run this script three times with separate configs (or override ``series_index``).
"""

# Stage 3

from swissphenocam.cli.transitions import main

if __name__ == "__main__":
    main()
