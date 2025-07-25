import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy import util
from arkouda.util import is_float, is_int, is_numeric, map


class TestUtil:
    def test_util_docstrings(self):
        import doctest

        result = doctest.testmod(util, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_sparse_sum_helper(self):
        cfg = ak.get_config()
        N = (10**4) * cfg["numLocales"]
        select_from = ak.arange(N)
        inds1 = select_from[ak.randint(0, 10, N) % 3 == 0]
        inds2 = select_from[ak.randint(0, 10, N) % 3 == 0]
        vals1 = ak.randint(-(2**32), 2**32, N)[inds1]
        vals2 = ak.randint(-(2**32), 2**32, N)[inds2]

        merge_idx, merge_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=True)
        sort_idx, sort_vals = ak.util.sparse_sum_help(inds1, inds2, vals1, vals2, merge=False)
        gb_idx, gb_vals = ak.GroupBy(ak.concatenate([inds1, inds2], ordered=False)).sum(
            ak.concatenate((vals1, vals2), ordered=False)
        )

        assert (merge_idx == sort_idx).all()
        assert (merge_idx == gb_idx).all()
        assert (merge_vals == sort_vals).all()

    def test_is_numeric(self):
        strings = ak.array(["a", "b"])
        ints = ak.array([1, 2])
        categoricals = ak.Categorical(strings)
        floats = ak.array([1, np.nan])

        from arkouda.pandas.index import Index
        from arkouda.pandas.series import Series

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            assert not is_numeric(item)

        for item in [
            ints,
            Index(ints),
            Series(ints),
            floats,
            Index(floats),
            Series(floats),
        ]:
            assert is_numeric(item)

        for item in [
            strings,
            Index(strings),
            Series(strings),
            categoricals,
            Index(categoricals),
            Series(categoricals),
            floats,
            Index(floats),
            Series(floats),
        ]:
            assert not is_int(item)

        for item in [ints, Index(ints), Series(ints)]:
            assert is_int(item)

        for item in [
            strings,
            Index(strings),
            Series(strings),
            ints,
            Index(ints),
            Series(ints),
            categoricals,
            Index(categoricals),
            Series(categoricals),
        ]:
            assert not is_float(item)

        for item in [floats, Index(floats), Series(floats)]:
            assert is_float(item)

    def test_map(self):
        a = ak.array(["1", "1", "4", "4", "4"])
        b = ak.array([2, 3, 2, 3, 4])
        c = ak.array([1.0, 1.0, 2.2, 2.2, 4.4])
        d = ak.Categorical(a)

        result = map(a, {"4": 25, "5": 30, "1": 7})
        assert result.tolist() == [7, 7, 25, 25, 25]

        result = map(a, {"1": 7})
        assert result.tolist() == ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).tolist()

        result = map(a, {"1": 7.0})
        assert np.allclose(result.tolist(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

        result = map(b, {4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        assert result.tolist() == [30.0, 5.0, 30.0, 5.0, 25.0]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d"})
        assert result.tolist() == ["a", "a", "b", "b", "c"]

        result = map(c, {1.0: "a"})
        assert result.tolist() == ["a", "a", "null", "null", "null"]

        result = map(c, {1.0: "a", 2.2: "b", 4.4: "c", 5.0: "d", 6.0: "e"})
        assert result.tolist() == ["a", "a", "b", "b", "c"]

        result = map(d, {"4": 25, "5": 30, "1": 7})
        assert result.tolist() == [7, 7, 25, 25, 25]

        result = map(d, {"1": 7})
        assert np.allclose(
            result.tolist(),
            ak.cast(ak.array([7, 7, np.nan, np.nan, np.nan]), dt=ak.int64).tolist(),
            equal_nan=True,
        )

        result = map(d, {"1": 7.0})
        assert np.allclose(result.tolist(), [7.0, 7.0, np.nan, np.nan, np.nan], equal_nan=True)

    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.bool_, ak.bigint, ak.str_])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_copy(self, dtype, size):
        a = ak.arange(size, dtype=dtype)
        b = ak.numpy.util.copy(a)

        from arkouda import assert_equal as ak_assert_equal

        assert a is not b
        ak_assert_equal(a, b)
