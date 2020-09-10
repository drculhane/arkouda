module GenSymIO {
    use HDF5;
    use IO;
    use Path;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use FileSystem;
    use Sort;
    use CommAggregation;
    use NumPyDType;
    use List;
    use Map;
    use PrivateDist;

    config const GenSymIO_DEBUG = false;
    config const SEGARRAY_OFFSET_NAME = "segments";
    config const SEGARRAY_VALUE_NAME = "values";
    config const NULL_STRINGS_VALUE = 0:uint(8);
    config const TRUNCATE: int = 0;
    config const APPEND: int = 1;

    /*
     * Creates a pdarray server-side and returns the SymTab name used to
     * retrieve the pdarray from the SymTab.
     */
    proc arrayMsg(cmd: string, payload: bytes, st: borrowed SymTab): string {
        var repMsg: string;
        var (dtypeBytes, sizeBytes, data) = payload.splitMsgToTuple(3);
        var dtype = str2dtype(try! dtypeBytes.decode());
        var size = try! sizeBytes:int;
        var tmpf:file;

        // Write the data payload composing the pdarray to a memory buffer
        try {
            tmpf = openmem();
            var tmpw = tmpf.writer(kind=iobig);
            tmpw.write(data);
            try! tmpw.close();
        } catch {
            return "Error: Could not write to memory buffer";
        }

        // Get the next name from the SymTab cache
        var rname = st.nextName();

        /*
         * Read the data payload from the memory buffer, encapsulate
         * within a SymEntry, and write to the SymTab cache  
         */
        try {
            var tmpr = tmpf.reader(kind=iobig, start=0);
            if dtype == DType.Int64 {
                var entryInt = new shared SymEntry(size, int);
                tmpr.read(entryInt.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryInt);
            } else if dtype == DType.Float64 {
                var entryReal = new shared SymEntry(size, real);
                tmpr.read(entryReal.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryReal);
            } else if dtype == DType.Bool {
                var entryBool = new shared SymEntry(size, bool);
                tmpr.read(entryBool.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryBool);
            } else if dtype == DType.UInt8 {
                var entryUInt = new shared SymEntry(size, uint(8));
                tmpr.read(entryUInt.a);
                tmpr.close(); tmpf.close();
                st.addEntry(rname, entryUInt);
            } else {
                tmpr.close();
                tmpf.close();
                return try! "Error: Unhandled data type %s".format(dtypeBytes);
            }
            tmpr.close();
            tmpf.close();
        } catch {
            return "Error: Could not read from memory buffer into SymEntry";
        }
        /*
         * Return message indicating the SymTab name corresponding to the
         * newly-created pdarray
         */
        return try! "created " + st.attrib(rname);
    }

    /*
     * Outputs the pdarray as a Numpy ndarray in the form of a 
     * Chapel Bytes object
     */
    proc tondarrayMsg(cmd: string, payload: bytes, st: 
                                          borrowed SymTab): bytes throws {
        var arrayBytes: bytes;
        var entryStr = payload.decode();
        var entry = st.lookup(entryStr);
        var tmpf: file;
        try {
            tmpf = openmem();
            var tmpw = tmpf.writer(kind=iobig);
            if entry.dtype == DType.Int64 {
                tmpw.write(toSymEntry(entry, int).a);
            } else if entry.dtype == DType.Float64 {
                tmpw.write(toSymEntry(entry, real).a);
            } else if entry.dtype == DType.Bool {
                tmpw.write(toSymEntry(entry, bool).a);
            } else if entry.dtype == DType.UInt8 {
                tmpw.write(toSymEntry(entry, uint(8)).a);
            } else {
                return try! b"Error: Unhandled dtype %s".format(entry.dtype);
            }
            tmpw.close();
        } catch {
            try! tmpf.close();
            return b"Error: Unable to write SymEntry to memory buffer";
        }

        try {
            var tmpr = tmpf.reader(kind=iobig, start=0);
            tmpr.readbytes(arrayBytes);
            tmpr.close();
            tmpf.close();
        } catch {
            return b"Error: Unable to copy array from memory buffer to string";
        }
        //var repMsg = try! "Array: %i".format(arraystr.length) + arraystr;
        /*
         Engin: fwiw, if you want to achieve the above, you can:

         return b"Array: %i %|t".format(arrayBytes.length, arrayBytes);

         But I think the main problem is how to separate the length from the data
         */
       return arrayBytes;
    }

    class DatasetNotFoundError: Error {proc init() {}}
    class NotHDF5FileError: Error {proc init() {}}
    class MismatchedAppendError: Error {proc init() {}}
    class WriteModeError: Error { proc init() {} }
    class SegArrayError: Error {proc init() {}}

    /*
     * Converts the JSON array to a pdarray
     */
    proc jsonToPdArray(json: string, size: int) throws {
        var f = opentmp();
        var w = f.writer();
        w.write(json);
        w.close();
        var r = f.reader(start=0);
        var array: [0..#size] string;
        r.readf("%jt", array);
        r.close();
        f.close();
        return array;
    }

    /*
     * Spawns a separate Chapel process that executes and returns the 
     * result of the h5ls command
     */
    proc lshdfMsg(cmd: string, payload: bytes,
                                st: borrowed SymTab): string throws {
        // reqMsg: "lshdf [<json_filename>]"
        use Spawn;
        const tmpfile = "/tmp/arkouda.lshdf.output";
        var repMsg: string;
        var (jsonfile) = payload.decode().splitMsgToTuple(1);

        var filename: string;
        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(1, jsonfile);
        }

        // Attempt to interpret filename as a glob expression and ls the first result
        var tmp = glob(filename);
        if GenSymIO_DEBUG {
            writeln(try! "glob expanded %s to %i files".format(filename, tmp.size));
        }
        if tmp.size <= 0 {
            return try! "Error: no files matching %s".format(filename);
        }
        filename = tmp[tmp.domain.first];
        var exitCode: int;
        try {
            if exists(tmpfile) {
                remove(tmpfile);
            }
            var cmd = try! "h5ls \"%s\" > \"%s\"".format(filename, tmpfile);
            var sub = spawnshell(cmd);
            // sub.stdout.readstring(repMsg);
            sub.wait();
            exitCode = sub.exit_status;
            var f = open(tmpfile, iomode.r);
            var r = f.reader(start=0);
            r.readstring(repMsg);
            r.close();
            f.close();
            remove(tmpfile);
        } catch {
            return "Error: failed to spawn process and read output";
        }

        if exitCode != 0 {
            return try! "Error: %s".format(repMsg);
        } else {
            return repMsg;
        }
    }

    /* Read dataset from HDF5 files into arkouda symbol table. */
    proc readhdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var repMsg: string;
        // reqMsg = "readhdf <dsetName> <nfiles> [<json_filenames>]"
        var (dsetName, nfilesStr, jsonfiles) = payload.decode().splitMsgToTuple(3);
        var nfiles = try! nfilesStr:int;
        var filelist: [0..#nfiles] string;
        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        }
        var filedom = filelist.domain;
        var filenames: [filedom] string;
        if filelist.size == 1 {
            var tmp = glob(filelist[0]);
            if GenSymIO_DEBUG {
                writeln(try! "glob expanded %s to %i files".format(filelist[0], tmp.size));
            }
            if tmp.size == 0 {
                return try! "Error: no files matching %s".format(filelist[0]);
            }
            // Glob returns filenames in weird order. Sort for consistency
            // sort(tmp);
            filedom = tmp.domain;
            filenames = tmp;
        } else {
            filenames = filelist;
        }

        var segArrayFlags: [filedom] bool;
        var dclasses: [filedom] C_HDF5.hid_t;
        var bytesizes: [filedom] int;
        var signFlags: [filedom] bool;
        for (i, fname) in zip(filedom, filenames) {
            try {
                (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype(fname, dsetName);
            } catch e: FileNotFoundError {
                return try! "Error: file not found: %s".format(fname);
            } catch e: PermissionError {
                return try! "Error: permission error on %s".format(fname);
            } catch e: DatasetNotFoundError {
                return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
            } catch e: NotHDF5FileError {
                return try! "Error: cannot open as HDF5 file %s".format(fname);
            } catch e: SegArrayError {
                return try! "Error: expected segmented array but could not find sub-datasets '%s' and '%s'".
                                                                   format(SEGARRAY_OFFSET_NAME, SEGARRAY_VALUE_NAME);
            } catch {
                // Need a catch-all for non-throwing function
                return try! "Error: unknown cause";
            }
        }
        const isSegArray = segArrayFlags[filedom.first];
        const dataclass = dclasses[filedom.first];
        const bytesize = bytesizes[filedom.first];
        const isSigned = signFlags[filedom.first];
        for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
            if (sa != isSegArray) || (dc != dataclass) || (bs != bytesize) || (sf != isSigned) {
                return try! "Error: inconsistent dtype in dataset %s of file %s".format(dsetName, name);
            }
        }
        if GenSymIO_DEBUG {
            writeln("Verified all dtypes across files");
        }
        var subdoms: [filedom] domain(1);
        var segSubdoms: [filedom] domain(1);
        var len: int;
        var nSeg: int;
        try {
            if isSegArray {
                (segSubdoms, nSeg) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                (subdoms, len) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
            } else {
                (subdoms, len) = get_subdoms(filenames, dsetName);
            }
        } catch e: HDF5RankError {
            return notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
        } catch {
            return try! "Error: unknown cause";
        }
        if GenSymIO_DEBUG {
            writeln("Got subdomains and total length");
        }

        select (isSegArray, dataclass) {
            when (true, C_HDF5.H5T_INTEGER) {
                if (bytesize != 1) || isSigned {
                    return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".
                                            format(isSegArray, dataclass, bytesize, isSigned);
                }
                var entrySeg = new shared SymEntry(nSeg, int);
                read_files_into_distributed_array(entrySeg.a, segSubdoms, filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                fixupSegBoundaries(entrySeg.a, segSubdoms, subdoms);
                var entryVal = new shared SymEntry(len, uint(8));
                read_files_into_distributed_array(entryVal.a, subdoms, filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);

                var segName = st.nextName();
                st.addEntry(segName, entrySeg);
                var valName = st.nextName();
                st.addEntry(valName, entryVal);
                return try! "created " + st.attrib(segName) + " +created " + st.attrib(valName);
            }
            when (false, C_HDF5.H5T_INTEGER) {
                var entryInt = new shared SymEntry(len, int);
                if GenSymIO_DEBUG {
                    writeln("Initialized int entry"); try! stdout.flush();
                }
                read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryInt);
                return try! "created " + st.attrib(rname);
            }
            when (false, C_HDF5.H5T_FLOAT) {
                var entryReal = new shared SymEntry(len, real);
                if GenSymIO_DEBUG {
                    writeln("Initialized float entry"); try! stdout.flush();
                }
                read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                var rname = st.nextName();
                st.addEntry(rname, entryReal);
                return try! "created " + st.attrib(rname);
            }
            otherwise {
                return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
            }
        }
    }

    /* 
     * Reads all datasets from 1..n HDF5 files into an Arkouda symbol table. 
     */
    proc readAllHdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        // reqMsg = "readAllHdf <ndsets> <nfiles> [<json_dsetname>] | [<json_filenames>]"
        var repMsg: string;
        // May need a more robust delimiter then " | "
        var (ndsetsStr, nfilesStr, arraysStr) = payload.decode().splitMsgToTuple(3);
        var (jsondsets, jsonfiles) = arraysStr.splitMsgToTuple(" | ",2);
        var ndsets = try! ndsetsStr:int;
        var nfiles = try! nfilesStr:int;
        var dsetlist: [0..#ndsets] string;
        var filelist: [0..#nfiles] string;
        try {
            dsetlist = jsonToPdArray(jsondsets, ndsets);
        } catch {
            return try! "Error: could not decode json dataset names via tempfile (%i files: %s)".format(ndsets, jsondsets);
        }
        try {
            filelist = jsonToPdArray(jsonfiles, nfiles);
        } catch {
            return try! "Error: could not decode json filenames via tempfile (%i files: %s)".format(nfiles, jsonfiles);
        }
        var dsetdom = dsetlist.domain;
        var filedom = filelist.domain;
        var dsetnames: [dsetdom] string;
        var filenames: [filedom] string;
        dsetnames = dsetlist;

        if filelist.size == 1 {
            var tmp = glob(filelist[0]);
            if GenSymIO_DEBUG {
                writeln(try! "glob expanded %s to %i files".format(filelist[0], tmp.size));
            }
            if tmp.size == 0 {
                return try! "Error: no files matching %s".format(filelist[0]);
            }
            // Glob returns filenames in weird order. Sort for consistency
            // sort(tmp);
            filedom = tmp.domain;
            filenames = tmp;
        } else {
            filenames = filelist;
        }
        var segArrayFlags: [filedom] bool;
        var dclasses: [filedom] C_HDF5.hid_t;
        var bytesizes: [filedom] int;
        var signFlags: [filedom] bool;
        var rnames: string;
        for dsetName in dsetnames do {
            for (i, fname) in zip(filedom, filenames) {
                try {
                    (segArrayFlags[i], dclasses[i], bytesizes[i], signFlags[i]) = get_dtype(fname, dsetName);
                } catch e: FileNotFoundError {
                    return try! "Error: file not found: %s".format(fname);
                } catch e: PermissionError {
                    return try! "Error: permission error on %s".format(fname);
                } catch e: DatasetNotFoundError {
                    return try! "Error: dataset %s not found in file %s".format(dsetName, fname);
                } catch e: NotHDF5FileError {
                    return try! "Error: cannot open as HDF5 file %s".format(fname);
                } catch e: SegArrayError {
                    return try! "Error: expected segmented array but could not find sub-datasets '%s' and '%s'".
                                          format(SEGARRAY_OFFSET_NAME, SEGARRAY_VALUE_NAME);
                } catch {
                    // Need a catch-all for non-throwing function
                    return try! "Error: unknown cause";
                }
            }
            const isSegArray = segArrayFlags[filedom.first];
            const dataclass = dclasses[filedom.first];
            const bytesize = bytesizes[filedom.first];
            const isSigned = signFlags[filedom.first];
            for (name, sa, dc, bs, sf) in zip(filenames, segArrayFlags, dclasses, bytesizes, signFlags) {
                if (sa != isSegArray) || (dc != dataclass) || (bs != bytesize) || (sf != isSigned) {
                    return try! "Error: inconsistent dtype in dataset %s of file %s".format(dsetName, name);
                }
            }
            if GenSymIO_DEBUG {
                writeln("Verified all dtypes across files for dataset ", dsetName);
            }
            var subdoms: [filedom] domain(1);
            var segSubdoms: [filedom] domain(1);
            var len: int;
            var nSeg: int;
            try {
                if isSegArray {
                    (segSubdoms, nSeg) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                    (subdoms, len) = get_subdoms(filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
                } else {
                    (subdoms, len) = get_subdoms(filenames, dsetName);
                }
            } catch e: HDF5RankError {
                return notImplementedError("readhdf", try! "Rank %i arrays".format(e.rank));
            } catch {
                return try! "Error: unknown cause";
            }
            if GenSymIO_DEBUG {
                writeln("Got subdomains and total length for dataset ", dsetName);
            }
            select (isSegArray, dataclass) {
                when (true, C_HDF5.H5T_INTEGER) {
                    if (bytesize != 1) || isSigned {
                        return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                    }
                    var entrySeg = new shared SymEntry(nSeg, int);
                    read_files_into_distributed_array(entrySeg.a, segSubdoms, filenames, dsetName + "/" + SEGARRAY_OFFSET_NAME);
                    fixupSegBoundaries(entrySeg.a, segSubdoms, subdoms);
                    var entryVal = new shared SymEntry(len, uint(8));
                    read_files_into_distributed_array(entryVal.a, subdoms, filenames, dsetName + "/" + SEGARRAY_VALUE_NAME);
                    var segName = st.nextName();
                    st.addEntry(segName, entrySeg);
                    var valName = st.nextName();
                    st.addEntry(valName, entryVal);
                    rnames = rnames + "created " + st.attrib(segName) + " +created " + st.attrib(valName) + " , ";
                }
                when (false, C_HDF5.H5T_INTEGER) {
                    var entryInt = new shared SymEntry(len, int);
                    if GenSymIO_DEBUG {
                        writeln("Initialized int entry for dataset ", dsetName); try! stdout.flush();
                    }
                    read_files_into_distributed_array(entryInt.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryInt);
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                when (false, C_HDF5.H5T_FLOAT) {
                    var entryReal = new shared SymEntry(len, real);
                    if GenSymIO_DEBUG {
                        writeln("Initialized float entry"); try! stdout.flush();
                    }
                    read_files_into_distributed_array(entryReal.a, subdoms, filenames, dsetName);
                    var rname = st.nextName();
                    st.addEntry(rname, entryReal);
                    rnames = rnames + "created " + st.attrib(rname) + " , ";
                }
                otherwise {
                    return try! "Error: detected unhandled datatype: segmented? %t, class %i, size %i, signed? %t".format(isSegArray, dataclass, bytesize, isSigned);
                }
            }
        }
        return try! rnames.strip(" , ", leading = false, trailing = true);
    }

    proc fixupSegBoundaries(a: [?D] int, segSubdoms: [?fD] domain(1), valSubdoms: [fD] domain(1)) {
        var boundaries: [fD] int; // First index of each region that needs to be raised
        var diffs: [fD] int;// Amount each region must be raised over previous region
        forall (i, sd, vd, b) in zip(fD, segSubdoms, valSubdoms, boundaries) {
            b = sd.low; // Boundary is index of first segment in file
            // Height increase of next region is number of bytes in current region
            if (i < fD.high) {
                diffs[i+1] = vd.size;
            }
        }
        // Insert height increases at region boundaries
        var sparseDiffs: [D] int;
        forall (b, d) in zip(boundaries, diffs) with (var agg = newDstAggregator(int)) {
            agg.copy(sparseDiffs[b], d);
        }
        // Make plateaus from peaks
        var corrections = + scan sparseDiffs;
        // Raise the segment offsets by the plateaus
        a += corrections;
    }

    /* Get the class of the HDF5 datatype for the dataset. */
    proc get_dtype(filename: string, dsetName: string) throws {
        const READABLE = (S_IRUSR | S_IRGRP | S_IROTH);
        if !exists(filename) {
            throw new owned FileNotFoundError();
        }
        if !(getMode(filename) & READABLE) {
            throw new owned PermissionError();
        }
        var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
        if file_id < 0 { // HF5open returns negative value on failure
            throw new owned NotHDF5FileError();
        }
        if !C_HDF5.H5Lexists(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT) {
            throw new owned DatasetNotFoundError();
        }
        var dataclass: C_HDF5.H5T_class_t;
        var bytesize: int;
        var isSigned: bool;
        var isSegArray: bool;

        try {
            (dataclass, bytesize, isSigned) = get_dataset_info(file_id, dsetName);
            isSegArray = false;
        } catch e:DatasetNotFoundError {
            var group_id = C_HDF5.H5Gopen2(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);
            if (group_id < 0) {
                try! writeln("The dataset is neither at the root of the HDF5 file nor within a group");
                throw new owned SegArrayError();
            }
            C_HDF5.H5Gclose(group_id);
            var offsetDset = dsetName + "/" + SEGARRAY_OFFSET_NAME;
            var valueDset = dsetName + "/" + SEGARRAY_VALUE_NAME;
            var (offsetClass, offsetByteSize, offsetSign) = try get_dataset_info(file_id, offsetDset);
            if (offsetClass != C_HDF5.H5T_INTEGER) {
                throw new owned SegArrayError();
            }
            try (dataclass, bytesize, isSigned) = get_dataset_info(file_id, valueDset);
            isSegArray = true;
        } catch e {
            throw e;
        }
        C_HDF5.H5Fclose(file_id);
        return (isSegArray, dataclass, bytesize, isSigned);
    }

    proc get_dataset_info(file_id, dsetName) throws {
        var dset = C_HDF5.H5Dopen(file_id, dsetName.c_str(), C_HDF5.H5P_DEFAULT);
        if (dset < 0) {
            throw new owned DatasetNotFoundError();
        }
        var datatype = C_HDF5.H5Dget_type(dset);
        var dataclass = C_HDF5.H5Tget_class(datatype);
        var bytesize = C_HDF5.H5Tget_size(datatype):int;
        var isSigned = (C_HDF5.H5Tget_sign(datatype) == C_HDF5.H5T_SGN_2);
        C_HDF5.H5Tclose(datatype);
        C_HDF5.H5Dclose(dset);
        return (dataclass, bytesize, isSigned);
    }

    class HDF5RankError: Error {
        var rank: int;
        var filename: string;
        var dsetName: string;
    }

    /*
     *  Get the subdomains of the distributed array represented by each file, 
     *  as well as the total length of the array. 
     */
    proc get_subdoms(filenames: [?FD] string, dsetName: string) throws {
        use SysCTypes;

        var lengths: [FD] int;
        for (i, filename) in zip(FD, filenames) {
            var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, C_HDF5.H5P_DEFAULT);
            var dims: [0..#1] C_HDF5.hsize_t; // Only rank 1 for now
//      var dsetRank: c_int;
//      // Verify 1D array
//      C_HDF5.H5LTget_dataset_ndims(file_id, dsetName.c_str(), dsetRank);
//      if dsetRank != 1 {
//        // TODO: change this to a throw
//        // halt("Expected 1D array, got rank " + dsetRank);
//        throw new owned HDF5RankError(dsetRank, filename, dsetName);
//      }
            // Read array length into dims[0]
            C_HDF5.HDF5_WAR.H5LTget_dataset_info_WAR(file_id, dsetName.c_str(), c_ptrTo(dims), nil, nil);
            C_HDF5.H5Fclose(file_id);
            lengths[i] = dims[0]: int;
        }
        // Compute subdomain of master array contained in each file
        var subdoms: [FD] domain(1);
        var offset = 0;
        for i in FD {
            subdoms[i] = {offset..#lengths[i]};
            offset += lengths[i];
        }
        return (subdoms, (+ reduce lengths));
    }

    /* This function gets called when A is a BlockDist or DefaultRectangular array. */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                                 filenames: [FD] string, dsetName: string)
        where (MyDmap == Dmap.blockDist || MyDmap == Dmap.defaultRectangular) {
            if GenSymIO_DEBUG {
                writeln("entry.a.targetLocales() = ", A.targetLocales()); try! stdout.flush();
                writeln("Filedomains: ", filedomains); try! stdout.flush();
            }
            coforall loc in A.targetLocales() do on loc {
                // Create local copies of args
                var locFiles = filenames;
                var locFiledoms = filedomains;
                var locDset = dsetName;
                /* On this locale, find all files containing data that belongs in
                 this locale's chunk of A */
                for (filedom, filename) in zip(locFiledoms, locFiles) {
                    var isopen = false;
                    var file_id: C_HDF5.hid_t;
                    var dataset: C_HDF5.hid_t;
                    // Look for overlap between A's local subdomains and this file
                    for locdom in A.localSubdomains() {
                        const intersection = domain_intersection(locdom, filedom);
                        if intersection.size > 0 {
                            // Only open the file once, even if it intersects with many local subdomains
                            if !isopen {
                                file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                                                                        C_HDF5.H5P_DEFAULT);
                                dataset = C_HDF5.H5Dopen(file_id, locDset.c_str(), C_HDF5.H5P_DEFAULT);
                                isopen = true;
                            }
                            // do A[intersection] = file[intersection - offset]
                            var dataspace = C_HDF5.H5Dget_space(dataset);
                            var dsetOffset = [(intersection.low - filedom.low): C_HDF5.hsize_t];
                            var dsetStride = [intersection.stride: C_HDF5.hsize_t];
                            var dsetCount = [intersection.size: C_HDF5.hsize_t];
                            C_HDF5.H5Sselect_hyperslab(dataspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(dsetOffset), 
                                                             c_ptrTo(dsetStride), c_ptrTo(dsetCount), nil);
                            var memOffset = [0: C_HDF5.hsize_t];
                            var memStride = [1: C_HDF5.hsize_t];
                            var memCount = [intersection.size: C_HDF5.hsize_t];
                            var memspace = C_HDF5.H5Screate_simple(1, c_ptrTo(memCount), nil);
                            C_HDF5.H5Sselect_hyperslab(memspace, C_HDF5.H5S_SELECT_SET, c_ptrTo(memOffset), 
                                                              c_ptrTo(memStride), c_ptrTo(memCount), nil);
                            if GenSymIO_DEBUG {
                                writeln("Locale ", loc, ", intersection ", intersection, ", dataset slice ", 
                                        (intersection.low - filedom.low, intersection.high - filedom.low));
                            }

                            /*
                             * The fact that intersection is a subset of a local subdomain means
                             * there should be no communication in the read
                             */
                            local {
                                C_HDF5.H5Dread(dataset, getHDF5Type(A.eltType), memspace, 
                                        dataspace, C_HDF5.H5P_DEFAULT, 
                                        c_ptrTo(A.localSlice(intersection)));
                            }
                            C_HDF5.H5Sclose(memspace);
                            C_HDF5.H5Sclose(dataspace);
                        }
                    }
                    if isopen {
                        C_HDF5.H5Dclose(dataset);
                        C_HDF5.H5Fclose(file_id);
                    }
                }
            }
        }

    /* This function is called when A is a CyclicDist array. */
    proc read_files_into_distributed_array(A, filedomains: [?FD] domain(1), 
                                           filenames: [FD] string, dsetName: string)
        where (MyDmap == Dmap.cyclicDist) {
            use CyclicDist;
            /*
             * Distribute filenames across locales, and ensure single-threaded
             * reads on each locale
             */
            var fileSpace: domain(1) dmapped Cyclic(startIdx=FD.low, dataParTasksPerLocale=1) = FD;
            forall fileind in fileSpace with (ref A) {
                var filedom: subdomain(A.domain) = filedomains[fileind];
                var filename = filenames[fileind];
                var file_id = C_HDF5.H5Fopen(filename.c_str(), C_HDF5.H5F_ACC_RDONLY, 
                                                                       C_HDF5.H5P_DEFAULT);
                // TODO: use select_hyperslab to read directly into a strided slice of A
                // Read file into a temporary array and copy into the correct chunk of A
                var AA: [1..filedom.size] A.eltType;
                readHDF5Dataset(file_id, dsetName, AA);
                A[filedom] = AA;
                C_HDF5.H5Fclose(file_id);
           }
    }

    proc domain_intersection(d1: domain(1), d2: domain(1)) {
        var low = max(d1.low, d2.low);
        var high = min(d1.high, d2.high);
        if (d1.stride !=1) && (d2.stride != 1) {
            //TODO: change this to throw
            halt("At least one domain must have stride 1");
        }
        var stride = max(d1.stride, d2.stride);
        return {low..high by stride};
    }

    proc tohdfMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        var (arrayName, dsetName, modeStr, jsonfile, dataType)
            = payload.decode().splitMsgToTuple(5);

        var mode = try! modeStr: int;
        var filename: string;
        var entry = st.lookup(arrayName);

        try {
            filename = jsonToPdArray(jsonfile, 1)[0];
        } catch {
            return try! "Error: could not decode json filenames via tempfile " +
                                                      "(%i files: %s)".format(1, jsonfile);
        }

        var warnFlag: bool;

        try {
            select entry.dtype {
                when DType.Int64 {
                    var e = toSymEntry(entry, int);
                    if isStringsSegmentsDataset(dsetName) {
                        warnFlag = write1DDistStrings(filename, mode, dsetName, e.a, DType.Int64);
                    } else {
                        warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Int64);
                    }
                }
                when DType.Float64 {
                    var e = toSymEntry(entry, real);
                    warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Float64);
                }
                when DType.Bool {
                    var e = toSymEntry(entry, bool);
                    warnFlag = write1DDistArray(filename, mode, dsetName, e.a, DType.Bool);
                }
                when DType.UInt8 {
                    var e = toSymEntry(entry, uint(8));
                    warnFlag = write1DDistStrings(filename, mode, dsetName, e.a, DType.UInt8);
                } otherwise {
                    return unrecognizedTypeError("tohdf", dtype2str(entry.dtype));
                }
            }
        } catch e: FileNotFoundError {
              return try! "Error: unable to open file for writing: %s".format(filename);
        } catch e: MismatchedAppendError {
              return "Error: appending to existing files must be done with the same number" +
                      "of locales. Try saving with a different directory or filename prefix?";
        } catch e: WriteModeError {
              return "Error: cannot append the non-existent file %s. Please save the file in standard truncate mode".format(filename);
        } catch e: Error {
              return "Error: problem writing to file %s".format(e);
        }
        if warnFlag {
            return "Warning: possibly overwriting existing files matching filename pattern";
        } else {
            return "wrote array to file";
        }
    }

    /*
     * Writes out the two pdarrays composing a Strings object to hdf5.
     */
    private proc write1DDistStrings(filename: string, mode: int, dsetName: string, A, 
                                                                array_type: DType) throws {
        var prefix: string;
        var extension: string;  
        var warnFlag: bool;      
        
        (prefix,extension) = getFileMetadata(filename);
 
        // Generate the filenames based upon the number of targetLocales.
        var filenames = generateFilenames(prefix, extension, A);
        
        //Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);
        
        var group = getGroup(dsetName); 
 
        if isStringsValuesDataset(dsetName) {
            warnFlag = processFilenames(filenames, matchingFilenames, mode, A, group);
        } else {
            warnFlag = false;
        }
        
        /*
         * The leadingSliceIndices object, which is a globally-scoped PrivateSpace 
         * array, contains the leading slice index for each locale, which is used 
         * to remove the uint(8) characters moved to the previous locale; this
         * situation occurs when a string spans two locales.
         * The trailingSliceIndices PrivateSpace is used in the special case 
         * where the majority of a large string spanning two locales is the sole
         * string on a locale; in this case, the trailing slice index is used
         * to move the smaller string chunk to the locale containing the large
         * string chunk that is the sole string chunk on a locale.
         */
        var leadingSliceIndices: [PrivateSpace] int;    
        var trailingSliceIndices: [PrivateSpace] int;
                  
        /*
         * The localeValuesIndexStart and localeValuesIndexEnd are globally-scoped
         * PrivateSpace arrays that represent the global start and end index for 
         * the Strings values chunk written to each hdf5 file. In other words, 
         * the start and end indices relative to the entire Strings values object 
         * as written to the collection of hdf5 files on 1..n locales.
         */
         var localeValuesIndexStart: [PrivateSpace] int;
         var localeValuesIndexEnd: [PrivateSpace] int;
           
           /*
            * The forwardShuffleSliceIndices is a globally-scoped PrivateSpace array
            * that index at which 1..n Strings segment elements are misassigned to the 
            * current locale and need to be re-assigned to the next locale.
            * 
            * :TODO: at some point, would be good to integrate an analogous array to
            * handle the situation where 1..n Strings segments elements are misassigned
            * to the next locale
            */
           var forwardShuffleSliceIndices: [PrivateSpace] int;

           /*
            * The localeValuesFileIndexEnd object is a globally-scoped PrivateSpace
            * array which contains the zero-based end index value for the Strings values 
            * array chunk written to hdf5. The localeValuesFileIndexEnd value is used
            * to generate localeValuesIndexStart and localeValuesIndexEnd values for
            * each locale.
            */
           var localeValuesFileIndexEnd: [PrivateSpace] int;

           /*
            * If this is a Strings values dataset, loop through all locales and set 
            * (1) leadingSliceIndices, which are used to remove leading uint(8) characters 
            * from the local slice that complete a string started in the previous locale and
            * (2) trailingSliceIndices, which are used to start strings that are completed in
            * the new locale.remove that belongs to the previous locale.
            */
           if isStringsValuesDataset(dsetName) {
               coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) 
                              with (ref leadingSliceIndices, ref trailingSliceIndices) do on loc {
                   const locDom = A.localSubdomain();
                   if idx < A.targetLocales().size-1 {
                       const locDom = A.localSubdomain();
                       if A.localSlice(locDom).back() != NULL_STRINGS_VALUE {
                           writeln("LOCALE %t STRINGS ARRAY DOES NOT END WITH NULL".format(idx));
                           generateSliceIndices(idx,leadingSliceIndices,trailingSliceIndices, A);
                       }
                   }
               }
           }

           /*
            * If this is a Strings segments dataset, there are three tasks for correctly assigning
            * the segments to each locale: (1) loop through all locales and set the Strings values
            * file end index, which is the zero-based index of uint(8) characters corresponding to
            * the Strings values chunk written to that file (2) calculate and assign the global 
            * start and end Strings values indices for each locale and (3) assign the forward
            * shuffle index needed to move miassigned segments elements from previous to the 
            * current locale.
            * 
            * The global Strings values indices assigned to each locale are used to shuffle Strings
            * segments per final assignment of the Strings values array elements to each locale 
            * and, consequently, each hdf5 file.
            */
           if isStringsSegmentsDataset(dsetName) {
               coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with (ref leadingSliceIndices,
                      ref localeValuesFileIndexEnd) do on loc {
                   if idx < A.targetLocales().size {
                       const fileName = filenames[idx];
                       if GenSymIO_DEBUG {
                           writeln(try! "%s exists? %t".format(fileName, exists(fileName)));
                       }

                       /*
                        * Retrieve the end index for the Strings values array chunk written 
                        * to the hdf5 file corresponding to the locale.
                        */
                       var fileId = C_HDF5.H5Fopen(fileName.c_str(), C_HDF5.H5F_ACC_RDWR,
                                                                       C_HDF5.H5P_DEFAULT);

                       var indexValues = getStringsValuesEndIndex(fileId, group);

                       /*
                        * Set the locale, and therefore hdf5 file-scoped Strings values 
                        * end index which will be used to build up the global Strings 
                        * segments start/end indices for each hdf5 file.
                        */
                       localeValuesFileIndexEnd[idx] = indexValues[0];

                       /*
                        * Close the file now that the Strings values end index has been 
                        * retrieved and assigned to the localeValuesFileIndexEnd array.
                        */
                        C_HDF5.H5Fclose(fileId);
                   }
              }

              /*
               * Loop through all of the locales, building up the global Strings values start
               * and end indices for each locale; these values will be used to shuffle segments
               * within each locale per the final assignment of Strings values array elements.
               * 
               * Note: the global values start and end indices need to be generated for each
               * locale independently to prevent race conditions in multilocale deployments. 
               */
              coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with (ref leadingSliceIndices,
                            ref localeValuesIndexStart, ref localeValuesIndexEnd, 
                      ref localeValuesFileIndexEnd, ref forwardShuffleSliceIndices) do on loc {
                  var start: int;
                  var end: int;
                  for i in 0..idx - 1 do {
                    start += localeValuesFileIndexEnd[i] + 1;
                  }
                  end = start + localeValuesFileIndexEnd[idx];

                  localeValuesIndexEnd[idx] = end;
                  localeValuesIndexStart[idx] = start;
                  forwardShuffleSliceIndices[idx] = -1;

                  const locDom = A.localSubdomain();
                  for (value, i) in zip(A.localSlice(locDom),
                                                    0..A.localSlice(locDom).size-1) {
                      if value >= localeValuesIndexEnd[idx] {
                          forwardShuffleSliceIndices[idx] = i;
                          break;
                      }
                  }
              }
           }
                                                       
        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare segments or values pdarray to be written (3) write pdarray to open
         * hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) with 
                        (ref leadingSliceIndices, ref trailingSliceIndices, ref localeValuesIndexStart, ref forwardShuffleSliceIndices,
                            ref localeValuesIndexEnd, ref localeValuesFileIndexEnd) do on loc {
            const myFilename = filenames[idx];
            if GenSymIO_DEBUG {
                writeln(try! "%s exists? %t".format(myFilename, exists(myFilename)));
            }
            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;
            var myDsetName = "/" + dsetName;

            use C_HDF5.HDF5_WAR;

          /*
           * A Strings values or segments dataset is handled differently because a string  
           * can span multiple locales since each string is composed of 1..n uint(8) 
           * characters. Accordingly, the first step in writing the local slice to hdf5 
           * is to verify if this is indeed a Strings values or segments dataset.
           */
          if isStringsValuesDataset(dsetName) {
            /*
             * Since this is Strings values array, confirm if it's in append mode. If
             * so, the Strings dataset is going to be appended to an hdf5 file as a 
             * set of values and segments arrays within a group named after the 
             * dsetName parameter.
             */
            if mode == APPEND {
                prepareStringsGroup(myFileID, group);
            }

            /*
             * Since this is a Strings values array, there is a possibility that 1..n
             * strings span two neighboring locales; this possibility is checked by
             * seeing if the final character in the local slice is the null uint(8)
             * character. If it is not, then the last string is only a partial string.
             */
            if A.localSlice(locDom).back() != NULL_STRINGS_VALUE {
              /*
               * Since the last value of the local slice is other than the uint(8) null
               * character, this means the last string in the current, local slice spans 
               * the current AND next locale. Consequently, need to do the following:
               * 1. Add all current locale slice values to a list
               * 2. Obtain remaining uint(8) values from the next locale
               */
              var charList : list(uint(8));
              var slicesList : list(int);
              (charList, slicesList) = convertLocalStringsSliceToList(A, locDom);
              writeln("LOCALE %t STRINGS VALUES CHARLIST %t DOES NOT END WITH NULL IS SIZE %t".format(idx, charList, charList.size));
              writeln("LOCALE %t slicesList %t".format(idx, slicesList));
              /*
               * On the next locale do the following:
               * 
               * 1. Retrieve the non-null uint(8) chars from the start of the local 
               *    slice until the next null uint(8) character is encountered
               * 2. Add to the newly-created charList
               */
              on Locales[idx+1] {
                  const locDom = A.localSubdomain();
                  var sliceList: list(uint(8), parSafe=true);

                  /*
                   * Iterate through the local slice values for the next locale and add
                   * each to the valuesList, until the null uint(8) character is reached.
                   * This subset of chars corresponds to the chars that complete the 
                   * last string of the previous locale (idx) if the
                   */
                   for (value, i) in zip(A.localSlice(locDom),
                                                  0..A.localSlice(locDom).size-1) {
                       if value != NULL_STRINGS_VALUE {
                           sliceList.append(value:uint(8));
                       } else {
                           break;
                       }
                   }

                   /*
                    * Check for an edge case where only one string maps to this locale, which is 
                    * indicated by the sliceList size matching the local slice array size; in 
                    * such a case there is only one segment.  If so, then keep the uint(0) chars
                    * here and shuffle chars from the previous locale to start the one and only
                    * string of the current locale.
                    */
                   if sliceList.size != A.localSlice(locDom).size {
                       charList.extend(sliceList);
                   } 
               }
              writeln("LOCALE %t STRINGS VALUES CHARLIST %t AFTER ACCOUNT FOR NULL NOW SIZE %t".format(idx, charList, charList.size));
               /* 
                * To prepare for writing revised values array to hdf5, do the following:
                * 1. Add null uint(8) char to the end of the array so reads work correctly
                * 2. Adjust the dims[0] value, which is the revised length of the valuesList
                */

               var leadingSliceIndex = leadingSliceIndices[idx]:int;
               var trailingSliceIndex : int; 
               
               if idx > 0 {
                   trailingSliceIndex = trailingSliceIndices[idx-1]:int;
               } else {
                   trailingSliceIndex = -1;
               }

               writeln("THE TRAILING INDEX FOR LOCALE %t IS %t".format(idx, trailingSliceIndices[idx]));
               var valuesList: list(uint(8), parSafe=true);

               /*
                * Now check to see if the current locale contains chars from the previous 
                * locale by checking the leadingSliceIndex. If the leadingSliceIndex > -1, this means 
                * that the charList contains chars that compose the last string from the previous 
                * locale. If so, generate a new valuesList that has those values sliced out.
                */
                if leadingSliceIndex > -1 {
                    (valuesList, slicesList) = adjustForLeadingSlice(leadingSliceIndex, charList);
                } else {
                    valuesList = charList;
                }
              
                /*
                 * Now check to see if the current locale contains chars that need to be shuffled to
                 * the next locale because the next locale has one string only. If so, then remove those
                 * characters, at which point the null uint(8) char is at the end of the values list.
                 * Otherwise, add the null uint(8) char to the end of the values list
                 */
                if trailingSliceIndex > -1 {
                    var sliceIndex = slicesList.last();
                    writeln("THE INDEX TO CHECK TAILING FOR %t: %t".format(idx,sliceIndex));
                    (valuesList, slicesList) = adjustForTrailingSlice(sliceIndex, valuesList);                
                } else {
                    valuesList.append(NULL_STRINGS_VALUE);            
                }
                
                // Update the dimensions per the possibly re-sized valuesList
                dims[0] = valuesList.size:uint(64);

                writeln("LOCALE %t CHARS LIST AFTER TRAILING SLICE ADJUSTMENT %t NOW SIZED %t".format(idx,valuesList,valuesList.size));
                writeln("THE SEGMENTS FOR LOCALE %t: %t".format(idx, slicesList));
                /*
                 * Write the valuesList containing the uint(8) characters missing from the
                 * current locale slice along with retrieved from the next locale to hdf5
                 */
                H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                        c_ptrTo(dims), getHDF5Type(A.eltType), c_ptrTo(valuesList.toArray()));

                /*
                 * Generate zero-based end index for values chunk written to hdf and write
                 * to the Strings hdf5 group; this value will be used to calculate the 
                 * global segments start and end indices that are used to shuffle Strings
                 * segments as needed to match Strings values and segments for each locale
                 * and, consequently, each hdf5 file.
                 */
                var valuesEndIndex = [valuesList.size-1];
                H5LTmake_dataset_WAR(myFileID, '/%s/values-index-bounds'.format(group).c_str(), 
                                    1, c_ptrTo([valuesEndIndex.size]), getHDF5Type(int),
                                    c_ptrTo(valuesEndIndex));
              } else {
                  /*
                   * The local slice ends with the uint(8) null character, which is the 
                   * required value to ensure correct read logic, so next check to see if 
                   * this local slice contains 1..n chars that compose a string from the 
                   * previous locale.
                   */
                  var leadingSliceIndex = leadingSliceIndices[idx]:int;

                  if leadingSliceIndex == -1 {
                      /*
                       * The local slice ends with the uint(8) null character, which means it's
                       * last string does not span two locales. Since the local slice also 
                       * not does not contain chars from previous locale, simply write the 
                       * Strings values slice out to hdf5.
                       */
                      H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                              c_ptrTo(dims), getHDF5Type(A.eltType), c_ptrTo(A.localSlice(locDom)));

                      /*
                       * Generate zero-based end index for values chunk written to hdf and write
                       * to the Strings hdf5 group; this value will be used to calculate the 
                       * global segments start and end indices that are used to shuffle Strings
                       * segments as needed to match Strings values and segments for each locale
                       * and, consequently, each hdf5 file.
                       */
                      var valuesEndIndex = [A.localSlice(locDom).size-1];
                      H5LTmake_dataset_WAR(myFileID, 
                              '/%s/values-index-bounds'.format(group).c_str(), 
                              1, c_ptrTo([[valuesEndIndex.size:uint(64)]]), getHDF5Type(int), 
                              c_ptrTo(valuesEndIndex));
                  } else {
                      /*
                       * The local slice does contain chars from previous locale, so (1)
                       * generate a corresponding Strings value list that can be sliced,
                       * and (2) adjust the Strings values list by slicing the chars out
                       * that correspond to chars from previous locale, and (3) adjust 
                       * the dims value per the size of the updated Strings value list. 
                       */
                       
                      var charList : list(uint(8));
                      var slicesList : list(int);
                      var valuesList : list(uint(8));
                     
                      (charList,slicesList) = convertLocalStringsSliceToList(A, locDom);                
                      (valuesList, slicesList) = adjustForLeadingSlice(leadingSliceIndex, charList);
                      writeln("LOCALE %t slicesList %t".format(idx, slicesList));
                      // Update the dimensions per the re-sized Strings values list
                      dims[0] = valuesList.size:uint(64);

                      H5LTmake_dataset_WAR(myFileID, '/%s/values'.format(group).c_str(), 1,
                                        c_ptrTo(dims), getHDF5Type(A.eltType),
                                        c_ptrTo(valuesList.toArray()));

                      /*
                       * Generate zero-based end index for values chunk written to hdf5 and write
                       * to the Strings hdf5 group; this value will be used to calculate the 
                       * global segments start and end indices that are used to shuffle Strings
                       * segments as needed to match Strings values and segments for each locale
                       * and, consequently, each hdf5 file.
                       */
                      var valuesEndIndex = [valuesList.size-1];
                      H5LTmake_dataset_WAR(myFileID, 
                                  '/%s/values-index-bounds'.format(group).c_str(), 
                                  1, c_ptrTo([valuesEndIndex.size:uint(64)]), getHDF5Type(int),
                                  c_ptrTo(valuesEndIndex));
                 }
            }
            } else if isStringsSegmentsDataset(dsetName) {
                /*
                 * Since this is a Strings segments pdarray, there are three situations that
                 * need to be accounted for: (1) ensure Strings segments elements are within the
                 * Strings values boundaries for the current locale (2) segments that are 
                 * misassigned to the previous locale (3) segments misassigned to the next locale. 
                 * 
                 * For the current locale, generate a segments list that contains elements that
                 * are within the String values range defined by the localeValuesIndexStart
                 * localeValuesIndexStart and localeValuesIndexEnd boundaries. 
                 * 
                 * Note: this situation occurs when Strings values elements are shuffled to 
                 * complete strings that span two locales.
                 */
                var newSegmentsList = generateBoundedStringsSegments(localeValuesIndexStart[idx], 
                        localeValuesIndexEnd[idx], A, locDom, idx);

                /*
                 * In terms of the previous locale, (applicable to locales 1..n), retrieve the 
                 * forwardShuffleSliceIndices element for the previous locale; if it is > -1, 
                 * this means that 1..n segments elements are incorrectly assigned to the wrong
                 * locale per the preceding Strings values reshuffle. Append newSegmentsList
                 * with all of the segments elements from that slice index and up.
                 *  
                 * Note: this situation occurs when String values elements are shuffled to 
                 * complete strings that span two locales.
                 */
                if idx > 0 {
                    var forwardShuffleSliceIndex = forwardShuffleSliceIndices[idx-1];
                    writeln("LOCALE %t FORWARD SHUFFLE INDEX IS %t".format(idx, forwardShuffleSliceIndex));
                    if forwardShuffleSliceIndex > -1 {
                        on Locales[idx-1] {
                            const locDom = A.localSubdomain();
                            for (value, i) in zip(A.localSlice(locDom),
                                                       0..A.localSlice(locDom).size-1) {
                                if i >= forwardShuffleSliceIndex {
                                    newSegmentsList.insert(0, value:int);
                                    writeln("LOCALE %t INSERTING FORWARD SHUFFLE SEGMENT %t".format(idx, value));
                                }
                            }
                        }
                    }
                }
                
                /*
                 * In terms of the next locale, iterate through that locale's Strings
                 * segments and, if any of those segments are outside that locale's 
                 * Strings values boundaries, append to this locale's segments list.
                 */
                if idx < filenames.size-1 {
                    on Locales[idx+1] {
                        const locDom = A.localSubdomain();
                        var stopIndex = localeValuesIndexStart[idx+1];

                        for (value, i) in zip(A.localSlice(locDom), 
                                                        0..A.localSlice(locDom).size-1) {
                            if value < stopIndex {
                                writeln("APPENDING %t FOR LOCALE %t".format(value,idx));
                                newSegmentsList.append(value:int);
                            } else {
                                break;
                            }
                        }
                    }
                }
                
                /*
                 * Sort the resulting newSegmentsList to ensure correct ordering of segments
                 * which is needed due to parallel processing in the on Locales statement blocks
                 */
                newSegmentsList.sort();

                /*
                 * Now that the segments are correctly assigned to the locale following adjustments
                 * per Strings values shuffling and ordered, rebase the segments list to a zero-based 
                 * array, which is needed to integrate into the Strings read framework. 
                 */
                var finalSegments = rebaseSegmentsDataset(newSegmentsList);
                H5LTmake_dataset_WAR(myFileID, '/%s/segments'.format(group).c_str(), 1,
                                           c_ptrTo([finalSegments.size:uint(64)]),getHDF5Type(int),
                                           c_ptrTo(finalSegments));
            }

            // Close the file now that the 1..n pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
        }
        return warnFlag;
    }

    private proc write1DDistArray(filename: string, mode: int, dsetName: string, A, 
                                                                 array_type: DType) throws {
        /* Output is 1 file per locale named <filename>_<loc>, and a dataset
        named <dsetName> is created in each one. If mode==1 (append) and the
        correct number of files already exists, then a new dataset named
        <dsetName> will be created in each. Strongly recommend only using
        append mode to write arrays with the same domain. */

        var prefix: string;
        var extension: string;
      
        (prefix,extension) = getFileMetadata(filename);

        // Generate the filenames based upon the number of targetLocales.
        var filenames = generateFilenames(prefix, extension, A);

        //Generate a list of matching filenames to test against. 
        var matchingFilenames = getMatchingFilenames(prefix, extension);

        var warnFlag = processFilenames(filenames, matchingFilenames, mode, A);

        /*
         * Iterate through each locale and (1) open the hdf5 file corresponding to the
         * locale (2) prepare pdarray(s) to be written (3) write pdarray(s) to open
         * hdf5 file and (4) close the hdf5 file
         */
        coforall (loc, idx) in zip(A.targetLocales(), filenames.domain) do on loc {
            const myFilename = filenames[idx];
            if GenSymIO_DEBUG {
                writeln(try! "%s exists? %t".format(myFilename, exists(myFilename)));
            }
            var myFileID = C_HDF5.H5Fopen(myFilename.c_str(), 
                                       C_HDF5.H5F_ACC_RDWR, C_HDF5.H5P_DEFAULT);
            const locDom = A.localSubdomain();
            var dims: [0..#1] C_HDF5.hsize_t;
            dims[0] = locDom.size: C_HDF5.hsize_t;
            var myDsetName = "/" + dsetName;

            use C_HDF5.HDF5_WAR;

            /*
             * This is neither a Strings values nor a Strings segments pdarray, so 
             * simply write the local slice out to the top-level group of the hdf5 file
             */
            H5LTmake_dataset_WAR(myFileID, myDsetName.c_str(), 1, c_ptrTo(dims),
                                      getHDF5Type(A.eltType), c_ptrTo(A.localSlice(locDom)));

            // Close the file now that the 1..n pdarrays have been written
            C_HDF5.H5Fclose(myFileID);
        }
        return warnFlag;
    }

    /*
     * Returns a tuple composed of a file prefix and extension to be used to generate
     * locale-specific filenames to be written to.
     */
    proc getFileMetadata(filename : string) {
        const fields = filename.split(".");
        var prefix: string;
        var extension: string;
 
        if fields.size == 1 || fields[fields.domain.high].count(pathSep) > 0 { 
            prefix = filename;
            extension = "";
        } else {
            prefix = ".".join(fields#(fields.size-1)); // take all but the last
            extension = "." + fields[fields.domain.high];
        }

        return (prefix,extension);
    }

    /*
     * Generates a list of filenames to be written to based upon a file prefix,
     * extension, and number of locales
     */
    proc generateFilenames(prefix : string, extension : string, A) : [] string { 
        // Generate the filenames based upon the number of targetLocales.
        var filenames: [0..#A.targetLocales().size] string;
        for i in 0..#A.targetLocales().size {
            filenames[i] = try! "%s_LOCALE%s%s".format(prefix, i:string, extension);
        }
        return filenames;
    }


    /*
     * If APPEND mode, checks to see if the matchingFilenames matches the filenames
     * array and, if not, raises a MismatchedAppendError. If in TRUNCATE mode, creates
     * the files matching the filenames. If 1..n of the filenames exist, returns 
     * warning to the user that 1..n files were overwritten. Since a group name is 
     * passed in, and hdf5 group is created in the file(s).
     */
    proc processFilenames(filenames: [] string, matchingFilenames: [] string, mode: int, 
                                            A, group: string) throws {
      // if appending, make sure number of files hasn't changed and all are present
      var warnFlag: bool;
      
      /*
       * Generate a list of matching filenames to test against. If in 
       * APPEND mode, check to see if list of filenames to be written
       * to match the names of existing files corresponding to the dsetName.
       * if in TRUNCATE mode, see if there are any filenames that match, 
       * meaning that 1..n files will be overwritten.
       */
      if (mode == APPEND) {
          var allexist = true;
          var anyexist = false;
          
          for f in filenames {
              var result =  try! exists(f);
              allexist &= result;
              if result {
                  anyexist = true;
              }
          }

          /*
           * Check to see if any exist. If not, this means the user is attempting to append
           * to 1..n files that don't exist. In this situation, the user is alerted that
           * the dataset must be saved in TRUNCATE mode.
           */
          if !anyexist {
              throw new owned WriteModeError();
          }

          /*
           * There is a mismatch between the number of files to be appended to and the 
           * number of files actually on the file system.
           */
          if !allexist || (matchingFilenames.size != filenames.size) {
              throw new owned MismatchedAppendError();
          }

      } else if mode == TRUNCATE { // if truncating, create new file per locale
          if matchingFilenames.size > 0 {
              warnFlag = true;
          } else {
              warnFlag = false;
          }

          for loc in 0..#A.targetLocales().size {
              /*
               * When done with a coforall over locales, only locale 0's file gets created
               * correctly, whereas hhe other locales' files have corrupted headers.
               */
              //filenames[loc] = try! "%s_LOCALE%s%s".format(prefix, loc:string, extension);
              var file_id: C_HDF5.hid_t;

              if GenSymIO_DEBUG {
                  writeln("Creating or truncating file");
              }

              file_id = C_HDF5.H5Fcreate(filenames[loc].c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                        C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
              
              prepareStringsGroup(file_id, group);

              if file_id < 0 { // Negative file_id means error
                  throw new owned FileNotFoundError();
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
           }
        } else {
            throw new IllegalArgumentError("The mode %t is invalid".format(mode));
        }    
        return warnFlag;
    }

    /*
     * If APPEND mode, checks to see if the matchingFilenams matches the filenames
     * array and, if not, raises a MismatchedAppendError. If in TRUNCATE mode, creates
     * the files matching the filenames. If 1..n of the filenames exist, returns 
     * warning to the user that 1..n files were overwritten.
     */
    proc processFilenames(filenames: [] string, matchingFilenames: [] string, mode: int, A) throws {
      // if appending, make sure number of files hasn't changed and all are present
      var warnFlag: bool;
      if (mode == APPEND) {
          var allexist = true;
          for f in filenames {
            allexist &= try! exists(f);
          }

          if !allexist || (matchingFilenames.size != filenames.size) {
              throw new owned MismatchedAppendError();
          }
      } else if mode == TRUNCATE { // if truncating, create new file per locale
          if matchingFilenames.size > 0 {
              warnFlag = true;
          } else {
              warnFlag = false;
          }

          for loc in 0..#A.targetLocales().size {
              /*
               * When done with a coforall over locales, only locale 0's file gets created
               * correctly, whereas hhe other locales' files have corrupted headers.
               */
              //filenames[loc] = try! "%s_LOCALE%s%s".format(prefix, loc:string, extension);
              var file_id: C_HDF5.hid_t;

              if GenSymIO_DEBUG {
                  writeln("Creating or truncating file");
              }

              file_id = C_HDF5.H5Fcreate(filenames[loc].c_str(), C_HDF5.H5F_ACC_TRUNC,
                                                        C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);

              if file_id < 0 { // Negative file_id means error
                  throw new owned FileNotFoundError();
              }

              /*
               * Close the file now that it has been created and, if applicable, the 
               * Strings group derived from the dsetName has been created.
               */
              C_HDF5.H5Fclose(file_id);
           }
        } else {
            throw new IllegalArgumentError("The mode %t is invalid".format(mode));
        }    
        return warnFlag;
    }
    
    /*
     * Generates an array of filenames to be matched in APPEND mode and to be
     * checked in TRUNCATE mode that will warn the user that 1..n files are
     * being overwritten.
     */
    proc getMatchingFilenames(prefix : string, extension : string) throws {
        return glob(try! "%s_LOCALE*%s".format(prefix, extension));    
    }

    /*
     * Generates the slice index for the locale Strings values array and adds it to the 
     * indices parameter. Note: the slice index will be used to remove characters from 
     * the current locale that correspond to the last string of the previous locale.
     */
    private proc generateSliceIndices(idx : int, leadingSliceIndices, trailingSliceIndices, A) {
        on Locales[idx+1] {
            const locDom = A.localSubdomain();
            var sliceIndex = -1;

            /*
             * Generate the leadlingSliceIndex that will filter out the non-null uint(8) 
             * characters, which are the characters that complete the last string started 
             * in the previous locale, along with the null uint(8) character. Consequently,
             * the resulting list will start at the first non-null uint(8) character,
             * which is the start of the first string to be assigned to the hdf5 file 
             * corresponding to this locale.
             */
            var leadingSliceSet = false;
            var leadingSliceIndex = -1;
            var trailingSliceIndex = -1;
            for (value, i) in zip(A.localSlice(locDom), 0..A.localSlice(locDom).size-1) {
                if value == NULL_STRINGS_VALUE {
                    if !leadingSliceSet {
                        /*
                         * Since the char is the null uint(8) character, that means that the
                         * chars composing the last string from the previous locale have been
                         * accounted for, so update the slice index and breakout from the for loop.
                         */
                        leadingSliceIndex = i + 1;
                        leadingSliceSet = true;
                        break;
                    } else {
                        trailingSliceIndex = i;
                    }
                }
            }

            leadingSliceIndices[here.id] = leadingSliceIndex;
            trailingSliceIndices[here.id] = trailingSliceIndex;
            try! writeln("FOR LOCALE %t GENERATED TSI %t".format(here.id, trailingSliceIndices[here.id]));
        }
    }

    /*
     * Converts a local Strings values slice into a uint(8) list for use in methods 
     * that add or remove entries from the resulting list.
     */
    private proc convertLocalStringsSliceToList(A, locDom) {
        var charList: list(uint(8), parSafe=true);
        var indices: list(int, parSafe=true);

        for (value, i) in zip(A.localSlice(locDom),
                                               0..A.localSlice(locDom).size-1) do {
            charList.append(value:uint(8));
            if value == NULL_STRINGS_VALUE {
                indices.append(i);            
            }

        }

        return (charList, indices);
    }

    /*
     * Converts a local segments slice into an int list for use in methods
     * that add or remove entries from the resulting list.
     */
    private proc convertLocalSegmentsSliceToList(A, locDom) : list(int) {
        var segmentList: list(int, parSafe=true);
        for segment in A.localSlice(locDom) {
            segmentList.append(segment:int);
        }
        return segmentList;
    }

    /*
     * Adjusts for the shuffling of a leading char sequence to the 
     * previous locale by (1) slicing leading chars that correspond
     * to 1..n chars composing a string started in the previous locale
     * and returning a new values list that composes all of the strings
     * that start in the current locale and (2) returns a new segments
     * list corresponding to the new values list
     */
    private proc adjustForLeadingSlice(sliceIndex : int,
                                   charList : list(uint(8))) {
        var valuesList: list(uint(8), parSafe=true);
        var indices: list(int);
        var i: int = 0;
        for value in charList(sliceIndex..charList.size-1)  {
            valuesList.append(value:uint(8));
            if value == NULL_STRINGS_VALUE {
                indices.append(i);
            }
            i+=1;
        }
        return (valuesList,indices);
    }

    /* 
     * Adjusts for the shuffling of a trailing char sequence to
     * the next locale by (1) slicing trailing chars that correspond
     * to 1..n chars composing a string that completes in the next
     * locale, returning a new list that composes all strings that
     * end in the current locale and (2) returns a new segments
     * list corresponding to the new values list
     */
    private proc adjustForTrailingSlice(sliceIndex : int,
                                   charList : list(uint(8))) {
        var valuesList: list(uint(8), parSafe=true);
        var indices: list(int);
        var i: int = 0;

        for value in charList(0..sliceIndex)  {
            valuesList.append(value:uint(8));
            if value == NULL_STRINGS_VALUE {
                indices.append(i);
            }
            i+=1;
        }
        
        if !indices.isEmpty() {
            indices.pop();
        }
        return (valuesList,indices);
    }

    /*
     * Adjusts the Strings segments by removing any segment values that are less 
     * than the Strings values array start index for the current locale. 
     * 
     * Note: this method is required to handle situations where the Strings values 
     * array is shuffled to the extent that 1..n of the segment values no longer 
     * correspond to a particular locale.
     */
    private proc generateBoundedStringsSegments(startIndex : int, endIndex : int, 
                                           A, locDom, idx) : list(int) {
        var newSegmentsList: list(int, parSafe=true);
        for segment in A.localSlice(locDom) {
            if segment >= startIndex && segment < endIndex {
                newSegmentsList.append(segment:int);
            }
        }
        return newSegmentsList;
    }

    /*
     * Reads the Strings values zero-based end index for a file, which corresponds to 
     * the values array chunk written to the file.
     */
    private proc getStringsValuesEndIndex(fileId: int, 
                                                   group: string) : [] int throws {
        var valuesLocaleIndices = [0];
        readHDF5Dataset(file_id=fileId, dsetName='/%s/values-index-bounds'.format(group), 
                             data=valuesLocaleIndices);
        return valuesLocaleIndices;
    }

    /*
     * Generates a list of segments, or indices to the start location
     * of each string within a uint(8) array. The segmentsList will be 
     * written to the hdf5 file as the segments dataset.
     * 
     * NOTE: this is deprecated as it was used in the first version of
     * Strings hdf5 file persistence
     */
    private proc generateSegmentsList(valuesList: list(uint(8))) : list(int) {
        var segmentsList: list(int, parSafe=true);

        /*
         * The valuesList starts with non-null uint(8) char, so
         * add zero index to mark the start of the first string
         */
        segmentsList.append(0);

        for (value, i) in zip(valuesList,0..valuesList.size-1) do {
            /*
             * If the value is a null uint(8) character, then the next
             * string sequence will start at the next list position,
             * so append sequences with current position + 1
             */
            if (value == NULL_STRINGS_VALUE) && (i < valuesList.size-1) {
                segmentsList.append(i+1);
            }
        }
        return segmentsList;
    }

    /*
     * Returns the name of the hdf5 group corresponding to a dataset name.
     */
    private proc getGroup(dsetName : string) : string throws {
        var rawGroupName = dsetName.split('STRINGS');
        var values = rawGroupName[1].split('/');
        if values.size < 1 {
            throw new IllegalArgumentError('The Strings dataset must be in form / {dset}/');
        } else {
            return values[1];
        }
    }

    /*
     * Creates an HDF5 Group named via the group parameter to store a String
     * object's segments and values pdarrays.
     * 
     * Note: The file corresponding to the fileId must be open prior to 
     * attempting the group create.
     */
    private proc prepareStringsGroup(fileId: int, group: string) throws {
        var groupId = C_HDF5.H5Gcreate2(fileId, "/%s".format(group).c_str(),
              C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT, C_HDF5.H5P_DEFAULT);
        C_HDF5.H5Gclose(groupId);
    }

    /*
     * Returns a boolean indicating whether the data set is a Strings 
     * values dataset corresponding to a Strings save operation.
     */
    private proc isStringsValuesDataset(dataset: string) : bool {
        if isStringsDataset(dataset) {
            return dataset.find(needle="values") > -1;
        } else {
            return false;
        }
    }

    /*
     * Returns a boolean indicating whether the data set is a Strings 
     * segments dataset corresponding to a Strings save operation.
     */
    private proc isStringsSegmentsDataset(dataset: string) : bool {
        if isStringsDataset(dataset) {
            return dataset.find(needle="segments") > -1;
        } else {
            return false;
        }
    }

    /*
     * Returns a boolean indicating whether the data set is a Strings values 
     * or Strings segments dataset corresponding to a Strings save operation.
     */
    private proc isStringsDataset(dataset: string) : bool {
        return dataset.find(needle="STRINGS") > -1;
    }

    /*
     * Resets the segments array values to start at zero, which is required if
     * the Strings.save is executed with number of locales > 1.
     */
    private proc rebaseSegmentsDataset(segments: list(int)) :  [] int {
        var dec : int;
        var newSegments: [0..segments.size-1] int;

        for (segment,i) in zip(segments,0..segments.size-1) do {
            if i == 0 {
                dec = segment:int;
            }
            newSegments[i] = segment:int - dec;
        }
        return newSegments;
    }
}