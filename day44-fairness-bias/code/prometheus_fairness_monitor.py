from prometheus_client import Gauge

disparity_index = Gauge('disparity_index', 'Bias across demographic groups')
disparity_index.set(dp_diff)
