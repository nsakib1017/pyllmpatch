"""Whole-file sweep operators recovered from the deterministic-repair research.

Vendored verbatim from the research scratch tree so the operators live in the repo and are
version-controlled, rather than being imported off a sys.path injection that would break the
moment the scratch directory is cleaned. Measured on the full malware population: these convert
30.3% of non-parsing files (11.1% touching zero lines, 19.2% via synthesized scaffolding).
"""
