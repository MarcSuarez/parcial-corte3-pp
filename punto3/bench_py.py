import numpy as np
import time
import tracemalloc
import resource

def train_linear(X, y, learning_rate=0.01, epochs=1000, verbose=True):
    w = 0.0
    b = 0.0
    m = len(X)

    tracemalloc.start()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        y_pred = w * X + b
        error = y_pred - y
        dw = (2/m) * np.dot(error, X)
        db = (2/m) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        if verbose and (epoch + 1) % 200 == 0:
            mse = np.mean(error ** 2)
            print(f"Epoch {epoch+1}, MSE: {mse:.6f}, w: {w:.6f}, b: {b:.6f}")
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"w": w, "b": b, "time_s": elapsed,
            "tracemalloc_peak_kb": peak//1024, "ru_maxrss_kb": ru}

X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)
res = train_linear(X, y, learning_rate=0.01, epochs=1000, verbose=True)
print("\nPython result:")
print(f"w ≈ {res['w']:.6f}, b ≈ {res['b']:.6f}")
print(f"time_s: {res['time_s']:.6f}s")
print(f"tracemalloc_peak_kb: {res['tracemalloc_peak_kb']} KB")
print(f"ru_maxrss_kb: {res['ru_maxrss_kb']} KB")

""""
Python result:
time_s: 0.044686s
tracemalloc_peak_kb: 1 KB
ru_maxrss_kb: 52672 KB

""""