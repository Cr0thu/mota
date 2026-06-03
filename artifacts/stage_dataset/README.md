# Stage Datasets

Generated stage datasets are not retained by default after cleanup. They can be regenerated from staged search routes when needed.

Example:

```bash
PYTHONPATH=src python -m mota_solver.solve_staged \
  --max-expansions-per-stage 5000 \
  --write-dataset \
  --dataset-out artifacts/stage_dataset/staged_5k.jsonl
```
