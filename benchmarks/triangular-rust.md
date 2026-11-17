# experiment/triangular-rust

Isolated benchmark experiment based on `port/v3.0.1`. No production backend is changed.

Run on the same machine and Python environment as the other experiments:

```bash
pip install -e '.[test,speed]' triangular>=0.1
python benchmarks/experiment_triangular.py
```

Accept only if the candidate wins at all representative sizes and passes the correctness checks in the script. Timings are medians after one warm-up; compare same-host runs.
