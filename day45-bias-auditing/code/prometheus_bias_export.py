from prometheus_client import Gauge

dp_diff = Gauge('dp_diff', 'Demographic Parity Difference')
dp_diff.set(0.07)
