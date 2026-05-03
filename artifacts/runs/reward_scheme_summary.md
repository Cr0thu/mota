# Reward Scheme Quick Evaluation

This is a lightweight simulator-only check. It does not train PPO; local optional RL deps are missing.

| scheme | replay reward | replay solved | greedy reward | greedy solved | greedy final | greedy hp/atk/def | greedy keys |
|---|---:|---|---:|---|---|---|---|
| raw | 1.101 | False | 0.395 | False | MT2:1,2 | 66/10/10 | Y0/B0/R0 |
| label_dense | 1.487 | False | 1.986 | False | MT5:2,11 | 4/10/10 | Y0/B1/R0 |
| milestone | 2.530 | False | 5.400 | False | MT3:10,11 | 240/11/10 | Y0/B1/R0 |
| resource_delta | 3.183 | False | 2.710 | False | MT3:2,11 | 1000/10/10 | Y5/B1/R0 |
| key_pressure | -12.207 | False | 12.190 | False | MT3:2,11 | 1000/10/10 | Y5/B1/R0 |
| potential | 0.976 | False | -0.295 | False | MT5:2,11 | 24/11/10 | Y2/B1/R0 |
