Commit Message: Optimize Python script for stock trading environment

Commit Description:

- Moved imports inside the methods where they are used.
- Precomputed values that are used repeatedly and stored them in variables to avoid repeated lookups.
- Used numpy vectorized operations wherever possible to improve performance.
