"""Classification core: normalization, matching and scoring.

Deliberately free of any user interface. The debug GUI and the command-line tool
that Power Automate calls are both thin wrappers over this package, so a
threshold tuned in the GUI is by construction the threshold the executable uses.
"""
