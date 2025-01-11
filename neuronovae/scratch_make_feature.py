import polars as pl
import numpy as np
from itertools import cycle
from boltons.iterutils import chunk_ranges


def make_feature(frames, interval) -> pl.DataFrame:
    num_features = 10
    feature_id = iter(cycle(range(10)))
    feature_encoding = np.zeros((num_features, frames), dtype=np.uint8)
    for indices in chunk_ranges(frames, interval):
        this_feature = next(feature_id)
        feature_encoding[this_feature, indices[0]:indices[1]] = 1
    return pl.DataFrame(feature_encoding.T, schema={f"{i}": pl.UInt8 for i in range(10)})
