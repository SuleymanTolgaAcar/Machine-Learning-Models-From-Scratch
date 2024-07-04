import numpy as np


class JohnsonsRule:

    def predict(self, df_jobs):
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
