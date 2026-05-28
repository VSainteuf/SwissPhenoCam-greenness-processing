#!/usr/bin/env python
"""Post-pipeline aggregation stub — delegates to swissphenocam.cli.aggregate.

The driver logic lives in ``src/swissphenocam/cli/aggregate.py`` so it can be
installed as a console script (``swissphenocam-aggregate``).
"""

from swissphenocam.cli.aggregate import main

if __name__ == "__main__":
    main()
