from __future__ import annotations

import pickle
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from toolz import first

from distributed.protocol.utils import pack_frames_prelude, unpack_frames

if TYPE_CHECKING:
    from pandas.core.internals.blocks import Block

try:
    # We'll never run a shuffle if numpy and pandas are not installed
    # TODO reinstate support for pandas 1.x
    import pandas as pd
    from pandas._libs.internals import BlockPlacement
    from pandas.core.arrays import StringArray
    from pandas.core.internals import BlockManager
    from pandas.core.internals.blocks import new_block
except Exception:
    pass


def pickle_bytelist(obj: object, prelude: bool = True) -> list[pickle.PickleBuffer]:
    """Variant of :func:`serialize_bytelist`, that doesn't support compression, locally
    defined classes, or any of its other fancy features but runs 10x faster for numpy
    arrays

    See Also
    --------
    serialize_bytelist
    unpickle_bytestream
    """
    frames: list = []
    pik = pickle.dumps(obj, protocol=5, buffer_callback=frames.append)
    frames.insert(0, pickle.PickleBuffer(pik))
    if prelude:
        frames.insert(0, pickle.PickleBuffer(pack_frames_prelude(frames)))
    return frames


def unpickle_bytestream(b: bytes | bytearray | memoryview) -> Iterator[Any]:
    """Unpickle the concatenated output of multiple calls to :func:`pickle_bytelist`

    See Also
    --------
    pickle_bytelist
    deserialize_bytes
    """
    while True:
        pik, *buffers, remainder = unpack_frames(b, remainder=True)
        yield pickle.loads(pik, buffers=buffers)
        if remainder.nbytes == 0:
            break
        b = remainder


def _bare_array(wrapper: Any) -> Any:
    if isinstance(wrapper, StringArray):
        # backend is ndarray[object] but has a __arrow_array__ method
        # TODO the wrapper class can be inferred from meta
        return StringArray, wrapper.__array__()  # type: ignore[attr-defined]
    elif hasattr(wrapper, "__arrow_array__"):
        # TODO the wrapper class can be inferred from the wrapped array
        return type(wrapper), wrapper.__arrow_array__()
    else:
        return wrapper.__array__()


def _rebuild_array(obj: Any) -> Any:
    if isinstance(obj, tuple):
        cls, values = obj
        return cls(values)
    else:
        return obj


def _bare_index(idx: pd.Index) -> tuple:
    if isinstance(idx, pd.MultiIndex):
        # Tiny ndarrays; faster to serialize as lists of ints
        codes = [c.tolist() for c in idx.codes]
        return *(_bare_index(level) for level in idx.levels), codes
    elif isinstance(idx, pd.RangeIndex):
        return idx.start, idx.stop, idx.step, idx.dtype, idx.name
    else:
        return _bare_array(idx.values), idx.name


def _rebuild_index(bare: tuple) -> pd.Index:
    if isinstance(bare[0], tuple):
        levels = [_rebuild_index(level) for level in bare[:-1]]
        codes = bare[-1]
        names = [level.name for level in levels]
        return pd.MultiIndex(levels, codes, names=names, verify_integrity=False)
    elif isinstance(bare[0], int):
        start, stop, step, dtype, name = bare
        return pd.RangeIndex(start, stop, step, dtype, name=name)
    else:
        values, name = bare
        return pd.Index(_rebuild_array(values), name=name)


def _bare_block(block: Block) -> tuple:
    arr = _bare_array(block.values)
    slice_ = block.mgr_locs.as_slice
    if slice_.stop == slice_.start + 1 and slice_.step == 1 and block.ndim == 1:
        return arr, slice_.start
    else:
        return arr, slice_.start, slice_.stop, slice_.step, block.ndim


def _rebuild_block(bare: tuple) -> Iterator[Block]:
    if len(bare) == 2:
        values, start = bare
        stop, step, ndim = start + 1, 1, 1
    else:
        values, start, stop, step, ndim = bare

    return new_block(
        _rebuild_array(values),
        BlockPlacement(slice(start, stop, step)),
        ndim=ndim,
    )


def pickle_dataframe_shard(
    input_part_id: int,
    shard: pd.DataFrame,
) -> list[pickle.PickleBuffer]:
    """Optimized pickler for pandas Dataframes. DIscard all unnecessary metadata
    (like the columns header).

    Parameters:
        obj: pandas
    """
    return pickle_bytelist(
        (
            input_part_id,
            _bare_index(shard.index),
            *(_bare_block(block) for block in shard._mgr.blocks),
        ),
        prelude=False,
    )


def unpickle_and_concat_dataframe_shards(
    b: bytes | bytearray | memoryview, meta: pd.DataFrame
) -> pd.DataFrame:
    """Optimized unpickler for pandas Dataframes.

    Parameters
    ----------
    b:
        raw buffer, containing the concatenation of the outputs of
        :func:`pickle_dataframe_shard`, in arbitrary order
    meta:
        DataFrame header

    Returns
    -------
    Reconstructed output shard, sorted by input partition ID

    **Roundtrip example**

    >>> import random
    >>> import pandas as pd
    >>> from toolz import concat

    >>> df = pd.DataFrame(...)  # Input partition
    >>> meta = df.iloc[:0].copy()
    >>> shards = df.iloc[0:10], df.iloc[10:20], ...
    >>> frames = [pickle_dataframe_shard(i, shard) for i, shard in enumerate(shards)]
    >>> random.shuffle(frames)  # Simulate the frames arriving in arbitrary order
    >>> blob = bytearray(b"".join(concat(frames)))  # Simulate disk roundtrip
    >>> df2 = unpickle_and_concat_dataframe_shards(blob, meta)
    """
    parts = list(unpickle_bytestream(b))
    # [(input_part_id, index, *blocks), ...]
    parts.sort(key=first)
    shards = []
    for _, raw_idx, *raw_blocks in parts:
        axes = [meta.columns, _rebuild_index(raw_idx)]
        blocks = [_rebuild_block(block) for block in raw_blocks]
        mgr = BlockManager(blocks, axes, verify_integrity=False)
        shard = pd.DataFrame._from_mgr(mgr, axes)  # type: ignore[attr-defined]
        shards.append(shard)

    # Actually load memory-mapped buffers into memory and close the file
    # descriptors
    return pd.concat(shards, copy=True)
