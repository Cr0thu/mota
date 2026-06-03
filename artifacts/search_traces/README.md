# Search Traces

Large generated search traces are intentionally not kept in the working tree. They are reproducible and were removed during the 2026-05-23 cleanup because they occupied about 659MB.

Regenerate traces with commands such as:

```bash
PYTHONPATH=src python -m mota_solver.solve_staged \
  --max-expansions-per-stage 5000 \
  --trace-out artifacts/search_traces/staged_trace.jsonl \
  --trace-limit 2000
```
