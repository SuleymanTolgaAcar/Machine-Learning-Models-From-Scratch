import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "Machine 1": [5, 4, 8, 2, 6, 12],
        "Machine 2": [5, 3, 9, 7, 8, 15],
    },
    index=["A", "B", "C", "D", "E", "F"],
)


def johnsons_rule(df_jobs):
    df = df_jobs.copy()
    sequence = np.empty(len(df), dtype=str)
    l = 0
    r = len(df) - 1
    while not df.empty:
        min_val = df.min().min()
        for machine in df.columns[::-1]:
            job = df[df[machine] == min_val]
            if job.empty:
                continue
            job = job.index[0]
            if machine == df.columns[0]:
                sequence[l] = job
                l += 1
            else:
                sequence[r] = job
                r -= 1
            df.drop(job, inplace=True)
            break

    return list(sequence)


print(johnsons_rule(df))
