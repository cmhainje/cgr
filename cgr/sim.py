import numpy as np

from .snapshot import Snapshot, GalaxySnapshot, MultiGalaxySnapshot


class Sim:
    def __init__(self, filepath):
        self.filepath = filepath
        self.params = self._read_paramfile()
        self.snapshot_filenames = self._find_snapshots()
        self.snapshot_numbers = self._make_snap_nums()
        self.snapshots = self._open_snapshots()
        self.times = self._read_times()

    def _read_paramfile(self):
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        params = {}
        for line in lines:
            if line.startswith("%"):
                continue

            chunks = line.split()
            if len(chunks) <= 1:
                continue

            k, v = chunks[:2]
            params[k] = v

        return params

    def _find_snapshots(self):
        from glob import glob
        from os.path import join, dirname

        # find the output directory
        output_dir = self.params["OutputDir"]
        if output_dir[0] != "/":
            output_dir = join(dirname(self.filepath), output_dir)

        # look for snapshots
        snapshot_base = self.params.get("SnapshotFileBase", "snapshot")
        return sorted(glob(join(output_dir, f"{snapshot_base}_*.hdf5")))

    def _make_snap_nums(self):
        from os.path import basename

        def _extract_num(fname):
            return int(basename(fname).split("_")[-1].split(".")[0])

        return [_extract_num(f) for f in self.snapshot_filenames]

    def _open_snapshots(self):
        return [Snapshot(fname) for fname in self.snapshot_filenames]

    def _read_times(self):
        return np.array([
            s.header()['Time'] for s in self.snapshots
        ])

    def __get__(self, index):
        """ints are treated as indices, strings are treated as snapshot numbers."""
        if isinstance(index, int):
            # return this index
            return self.snapshots[index]

        elif isinstance(index, str) and index.isdigit():
            # look for this snapshot number
            snap_num = int(index)
            index = self.snapshot_numbers.index(snap_num)
            return self.snapshots[index]

        else:
            raise ValueError(f"don't know what to do with this: {index}")


class GalaxySim(Sim):
    def __init__(self, filepath, drift_part_types=True):
        self.drift_pts = drift_part_types
        super().__init__(filepath)

    def _open_snapshots(self):
        return [
            GalaxySnapshot(fname, self.drift_pts) for fname in self.snapshot_filenames
        ]


class MultiGalaxySim(Sim):
    def __init__(self, filepath, infer_galaxies=True, drift_part_groups="host*"):
        self.drift_pgs = drift_part_groups
        self.infer = infer_galaxies
        super().__init__(filepath)

    def _open_snapshots(self):
        return [
            MultiGalaxySnapshot(fname, self.infer, self.drift_pgs)
            for fname in self.snapshot_filenames
        ]
