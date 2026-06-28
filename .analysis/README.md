# .analysis/ — scratch space for one-off log analysis

Gitignored. Drop throwaway analysis scripts, plots, and notes here so they
don't clutter the repo or `tools/`. Promote anything reusable into
`tools/dieslog.py` / `tools/match_analytics.py`.

Run scripts with the repo's venv, which already has pyarrow/pandas/matplotlib:

```bash
.venv/bin/python .analysis/my_script.py
```

Import the log readers from `tools/`:

```python
import sys; sys.path.insert(0, "tools")
from dieslog import load
log = load("logs/dies-2026-06-27_14-45-58")   # dir or .dieslog zip
```

See the "Log analysis" section of `CLAUDE.md` for the format + recipes.
