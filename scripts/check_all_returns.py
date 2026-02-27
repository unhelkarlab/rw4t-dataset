import sys
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python stats_npy_2d.py path/to/array.npy")
        sys.exit(1)

    path = sys.argv[1]
    arr = np.load(path)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got shape {arr.shape} (ndim={arr.ndim})")

    mean = arr.mean(axis=0)
    std_pop = arr.std(axis=0, ddof=0)  # population std
    std_samp = arr.std(axis=0, ddof=1)  # sample std

    print(f"Loaded: {path}")
    print(f"shape = {arr.shape}")

    # Pretty print per-column: mean ± std
    for j, (m, s0, s1) in enumerate(zip(mean, std_pop, std_samp)):
        print(
            f"col {j:02d}: mean={m:.2f} | std(ddof=0)={s0:.2f} | std(ddof=1)={s1:.2f} | {m:.2f} ± {s1:.2f}"
        )


if __name__ == "__main__":
    main()
