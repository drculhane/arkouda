module SortMsg
{
    use ServerConfig;

    use Time;
    use Math only;
    use ArkoudaSortCompat only relativeComparator;
    private use DynamicSort;
    use Search only;
    use Reflection;
    use ServerErrors;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use RadixSortLSD;
    use AryUtil;
    use Logging;
    use Message;
    private use ArgSortMsg;
    use NumPyDType only whichDtype;
    use BigInteger;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sortLogger = new Logger(logLevel, logChannel);

    /* Sort the given pdarray using Radix Sort and
       return sorted keys as a block distributed array */
    proc sort(a: [?aD] ?t): [aD] t throws {
      var sorted: [aD] t = radixSortLSD_keys(a);
      return sorted;
    }

    /* sort takes pdarray and returns a sorted copy of the array */
    @arkouda.registerCommand
    proc sort(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where ((t == real) || (t == int) || (t == uint(64)))
        do return sortHelp(array, alg, axis);

    proc sortHelp(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where d.rank == 1
    {
      var algorithm: SortingAlgorithm = ArgSortMsg.getSortingAlgorithm(alg);
      const itemsize = dtypeSize(whichDtype(t));
      overMemLimit(radixSortLSD_keys_memEst(d.size, itemsize));

      if algorithm == SortingAlgorithm.TwoArrayRadixSort {
        var sorted = makeDistArray(array);
        DynamicSort.dynamicTwoArrayRadixSort(sorted, comparator=myDefaultComparator);
        return sorted;
      } else {
        var sorted = radixSortLSD_keys(array);
        return sorted;
      }
    }

    proc sortHelp(array: [?d] ?t, alg: string, axis: int): [d] t throws
      where d.rank > 1
    {
      var algorithm: SortingAlgorithm = ArgSortMsg.getSortingAlgorithm(alg);
      const itemsize = dtypeSize(whichDtype(t));
      overMemLimit(radixSortLSD_keys_memEst(d.size, itemsize));

      const DD = domOffAxis(d, axis);
      var sorted = makeDistArray((...d.shape), t);

      if algorithm == SortingAlgorithm.TwoArrayRadixSort {
        for idx in DD {
          // make a copy of the array along the slice corresponding to idx
          // TODO: create a twoArrayRadixSort that operates on a slice of the array
          // in place instead of requiring the copy in/out
          var slice = makeDistArray(d.dim(axis).size, t);
          forall i in d.dim(axis) with (var perpIdx = idx) {
            perpIdx[axis] = i;
            slice[i] = array[perpIdx];
          }

          DynamicSort.dynamicTwoArrayRadixSort(slice, comparator=myDefaultComparator);

          forall i in d.dim(axis) with (var perpIdx = idx) {
            perpIdx[axis] = i;
            sorted[perpIdx] = slice[i];
          }
        }
      } else {
        // TODO: make a version of radixSortLSD_keys that does the sort on
        // slices of `e.a` directly instead of requiring a copy for each slice
        for idx in DD {
          const sliceDom = domOnAxis(d, idx, axis),
                sliced1D = removeDegenRanks(array[sliceDom], 1),
                sliceSorted = radixSortLSD_keys(sliced1D);

          forall i in sliceDom do sorted[i] = sliceSorted[i[axis]];
        }
      }

      return sorted;
    }

    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.searchsorted.html#array_api.searchsorted
    @arkouda.registerCommand
    proc searchSorted(x1: [?d1] ?t, x2: [?d2] t, side: string): [d2] int throws
      where ((d1.rank == 1) &&
             (t == int || t == real || t == uint || t == uint(8) || t == bigint))
    {
      if side != "left" && side != "right" {
          throw new Error("searchSorted side must be a string with value 'left' or 'right'.");
      }

      var ret = makeDistArray((...x2.shape), int);

      proc doSearch(const ref a1: [] t, const ref a2: [?d] t, cmp) {
        forall idx in ret.domain {
          const (_, i) = Search.binarySearch(a1, a2[idx], cmp);
          ret[idx] = i;
        }
      }

      select side {
        when "left" do doSearch(x1, x2, new leftCmp());
        when "right" do doSearch(x1, x2, new rightCmp());
        otherwise do halt("unreachable");
      }

      return ret;
    }

    record leftCmp: relativeComparator {
      proc compare(a: ?t, b: t): int
        where (t == int || t == real || t == uint || t == uint(8) || t == bigint)
      {
        if a <= b then return -1;
        else return 1;
      }
    }

    record rightCmp: relativeComparator {
      proc compare(a: ?t, b: t): int
        where (t == int || t == real || t == uint || t == uint(8) || t == bigint)
      {
        if a < b then return -1;
        else return 1;
      }
    }
}// end module SortMsg
