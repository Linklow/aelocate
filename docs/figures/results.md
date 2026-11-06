# Location benchmark: steel plate, linear location (2 sensors, 1 m base)

Calibrated velocity: threshold 5289 m/s, AIC 5304 m/s.

## all

| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±25 mm) |
|---|---:|---:|---:|---:|---:|
| Standard ΔT (threshold) | 68.2% | 65.6 | 41.7 | 174.8 | 30.5% |
| AIC picker | 68.2% | 62.8 | 40.0 | 166.9 | 30.8% |
| Neural network | 68.2% | 10.2 | 7.9 | 21.2 | 64.1% |
| Neural network, pulser-only training (ablation) | 68.2% | 45.9 | 36.5 | 97.8 | 24.9% |

## inside base

| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±25 mm) |
|---|---:|---:|---:|---:|---:|
| Standard ΔT (threshold) | 69.6% | 63.6 | 17.0 | 185.9 | 35.8% |
| AIC picker | 69.6% | 59.9 | 16.6 | 180.9 | 36.3% |
| Neural network | 69.6% | 9.3 | 7.1 | 19.7 | 66.5% |
| Neural network, pulser-only training (ablation) | 69.6% | 45.4 | 35.3 | 98.6 | 26.2% |

## near sensor (<=10 cm, inside)

| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±25 mm) |
|---|---:|---:|---:|---:|---:|
| Standard ΔT (threshold) | 64.3% | 33.9 | 15.7 | 84.9 | 35.7% |
| AIC picker | 64.3% | 31.5 | 15.7 | 84.9 | 35.7% |
| Neural network | 64.3% | 9.9 | 7.8 | 20.2 | 60.4% |
| Neural network, pulser-only training (ablation) | 64.3% | 48.3 | 38.7 | 105.6 | 21.9% |

## mid-span

| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±25 mm) |
|---|---:|---:|---:|---:|---:|
| Standard ΔT (threshold) | 71.0% | 70.4 | 20.2 | 192.4 | 35.9% |
| AIC picker | 71.0% | 66.5 | 17.7 | 188.3 | 36.4% |
| Neural network | 71.0% | 9.2 | 7.0 | 19.6 | 68.0% |
| Neural network, pulser-only training (ablation) | 71.0% | 44.7 | 34.6 | 98.0 | 27.3% |

## outside base

| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±25 mm) |
|---|---:|---:|---:|---:|---:|
| Standard ΔT (threshold) | 63.3% | 73.0 | 71.4 | 136.9 | 12.3% |
| AIC picker | 63.3% | 73.5 | 71.7 | 137.5 | 12.0% |
| Neural network | 63.3% | 13.2 | 10.5 | 26.4 | 56.0% |
| Neural network, pulser-only training (ablation) | 63.3% | 47.9 | 41.6 | 95.2 | 20.2% |
