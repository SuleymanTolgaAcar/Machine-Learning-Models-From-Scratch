import numpy as np
import pandas as pd

from johnsons_rule import JohnsonsRule

df = pd.DataFrame(
    {
        "Machine 1": [5, 4, 8, 2, 6, 12],
        "Machine 2": [5, 3, 9, 7, 8, 15],
    },
    index=["A", "B", "C", "D", "E", "F"],
)

model = JohnsonsRule()
sequence = model.predict(df)

print(sequence)
