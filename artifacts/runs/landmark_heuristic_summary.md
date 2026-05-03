# Landmark Heuristic Experiments

| run | route len | solved | final | hp/atk/def | keys | note |
|---|---:|---|---|---|---|---|
| baseline_200k | 47 | False | MT7:7,11 | 726/20/10 | Y0/B1/R0 | baseline, previous 200k |
| landmark_v1_10k | 9 | False | MT4:11,10 | 926/10/10 | Y4/B1/R0 | over-preserved keys |
| landmark_v2_10k | 98 | False | MT9:10,2 | 88/22/21 | Y4/B0/R0 | reduced early key weight |
| landmark_v2_200k | 110 | False | MT9:10,2 | 71/22/21 | Y4/B0/R0 | v2 200k |
| landmark_v3_10k | 98 | False | MT9:6,2 | 304/21/20 | Y2/B0/R0 | reduced money + low HP penalty |
| landmark_v3_50k | 105 | False | MT9:6,2 | 354/22/20 | Y3/B0/R0 | current best |
| landmark_v4_10k | 92 | False | MT8:4,4 | 348/21/20 | Y4/B0/R0 | target-distance test |
| landmark_v4_50k | 97 | False | MT8:1,2 | 380/22/20 | Y4/B0/R0 | target-distance test |
