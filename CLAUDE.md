# audiovj-ai

## Dependency pinning

- Prefer PyPI release pins (`pkg==X.Y.Z`) over git-commit pins.
- If a git-commit pin is unavoidable (e.g. `madmom`, which has no usable PyPI release), use only SHAs the user has vetted (typically lifted verbatim from another known-good environment).
- **Never** silently swap one git SHA for another from the same repo. If a pinned SHA fails to build or has an API mismatch, surface the problem and propose a PyPI fallback — do not search for "a newer commit that might work."
