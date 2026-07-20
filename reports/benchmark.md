# vLLM Throughput Benchmark

Best config: `{"max_num_seqs": 4, "gpu_memory_utilization": 0.85, "prefetch_workers": 8, "images_per_sec": 0.0}`

| max_num_seqs | gpu_mem | prefetch | img/s | tok/s | peakGB | fail | ok |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 | 0.8 | 4 | 0.000 | 0.0 | 0.00 | 1.00 | False |
| 2 | 0.85 | 4 | 0.000 | 0.0 | 0.00 | 1.00 | False |
| 4 | 0.8 | 4 | 0.000 | 0.0 | 0.00 | 1.00 | False |
| 4 | 0.85 | 4 | 0.000 | 0.0 | 0.00 | 1.00 | False |
