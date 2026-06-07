# Legacy Magic Tower Window

This folder is the old `mota-with-window` visualizer moved into the main
project. It is kept only as a lightweight manual play / reproduction window.

Run:

```bash
python tools/mota_with_window_legacy/run_this.py
```

Included:

- manual action selection;
- floor view switching;
- one-step backtracking;
- map assets and the first-10-floor JSON data.

Removed from this copy:

- old local learning modules and automated policy demos;
- trained model checkpoints;
- Python cache files and local editor settings.

For the maintained visualizer with route replay and real-time reward display,
use `tools/visualizer/run_this.py`.
