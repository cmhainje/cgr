import numpy as np
import h5py


class Snapshot:
    """A simple snapshot reader.

    Provides a simple and Pythonic interface to the data in the snapshot. The
    Snapshot class does not eagerly load any data from disk, nor does it keep
    an open h5py.File object around, so Snapshots are lightweight and can be
    created in any number.

    Helper functions are provided for a few common datasets: ParticleIDs,
    Coordinates, Velocities, and Masses. These are called `get_ids`, `get_pos`,
    `get_vel`, and `get_mass`. By default, these all sort the datasets by
    particle ID, since the actual ordering of particle data can vary between
    snapshots in some software. There are also helper functions that provide a
    simple interface to fetching other datasets: `get` and `get_sorted`.

    Datasets can be fetched from the snapshot by simple indexing. Indexing with
    the name of a dataset will load the dataset into memory as a NumPy array.
    Note that this differs from h5py, where indexing into an h5py.File with the
    name of a dataset will return a h5py.Dataset object that is only accessible
    as long as the File object is open and must be indexed a second time to
    load data into memory.

    With h5py, the second indexing that loads the data can also be used to load
    a subset of a dataset into memory (without needing to load an entire
    dataset at once). For an h5py.File object `f`, this might look like
    `f['data'][:10]`, which loads only the first ten entries of `'data'` into
    memory. To replicate this behavior with a Snapshot, pass the additional
    indexing in with the key as a tuple, like this: `s['data', :10]`. Note that
    `s['data'][:10]` will produce the same _result_, but it will do so by
    loading the entire dataset into memory, then applying the indexing in
    memory as a NumPy operation. For more complex indexing, e.g. for multiple
    dimensions, you can do either `s['data',:10,:5]` or `s['data',(:10,:5)]`.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __repr__(self):
        return "Snapshot(%r)" % (self.filepath)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            ind = key[1:]
            key = key[0]
            if len(ind) == 1:
                ind = ind[0]
        else:
            ind = slice(None)

        with h5py.File(self.filepath, "r") as f:
            return f[key][ind]

    def get_attrs(self, key):
        """Returns the attributes of an HDF5 Group as a dict."""
        with h5py.File(self.filepath, "r") as f:
            return dict(f[key].attrs)

    def config(self):
        """Returns the 'Config' attributes as a dict."""
        return self.get_attrs("Config")

    def header(self):
        """Returns the 'Header' attributes as a dict."""
        return self.get_attrs("Header")

    def _many_or_single(self, single_func, part_types, *args, **kwargs):
        if not isinstance(part_types, (tuple, list)):
            part_types = [part_types]
        if len(part_types) == 1:
            return single_func(part_types[0], *args, **kwargs)
        return np.concatenate(
            [single_func(pt, *args, **kwargs) for pt in part_types], axis=0
        )

    def get(self, part_type, key):
        """Returns the specified dataset(s).

        Fetches the dataset(s) associated with PartType(s) `part_type` that are
        named `key`. Note that this method can return the datasets for multiple
        particle types at once if `part_type` is given as a list or tuple or
        particle types. The results are concatenated and returned in the same
        order as given.

        Args:
            part_type: An int or string specifying a particle type, or a
                tuple/list of the same.
            key: A string specifying the dataset to return.

        Returns:
            A numpy array containing the requested dataset(s).
        """

        def _get(pt, key):
            return self[f"PartType{pt}/{key}"]

        return self._many_or_single(_get, part_type, key)

    def get_sorted(self, part_type, key):
        """Returns the specified dataset(s) sorted by particle ID.

        Fetches the dataset(s) associated with PartType(s) `part_type` that are
        named `key`. It then sorts the entries in this dataset according to
        particle IDs, so that the ordering is stable between snapshots.

        Note that this method can return the datasets for multiple particle
        types at once if `part_type` is given as a list or tuple or particle
        types. Each particle type's dataset is sorted individually; the results
        are concatenated and returned in the same order as given.

        Args:
            part_type: An int or string specifying a particle type, or a
                tuple/list of the same.
            key: A string specifying the dataset to return.

        Returns:
            A numpy array containing the requested dataset(s).
        """

        def _get_sorted(pt, key):
            ids = self[f"PartType{pt}/ParticleIDs"]
            val = self[f"PartType{pt}/{key}"]
            sort_indices = np.argsort(ids)
            return val[sort_indices]

        return self._many_or_single(_get_sorted, part_type, key)

    def get_ids(self, part_type):
        """Returns the (sorted) ParticleIDs dataset for `part_type`."""
        return self.get_sorted(part_type, "ParticleIDs")

    def get_pos(self, part_type):
        """Returns the (sorted) Coordinates dataset for `part_type`."""
        return self.get_sorted(part_type, "Coordinates")

    def get_vel(self, part_type):
        """Returns the (sorted) Velocities dataset for `part_type`."""
        return self.get_sorted(part_type, "Velocities")

    def get_mass(self, part_type):
        """Returns the (sorted) Masses dataset for `part_type`."""
        return self.get_sorted(part_type, "Masses")


class GalaxySnapshot(Snapshot):
    """Extension of the Snapshot class for isolated galaxy sims.

    The GalaxySnapshot computes the drift in the center of mass position and
    velocity of the isolated galaxy and automatically subtracts it from the
    results of `get_pos` and `get_vel`. One can specify only certain particle
    types to use when computing the drift via the drift_part_types argument. If
    True, all particle types in the snapshot are used. If False, the drifts are
    set to (0, 0, 0) and no computation is performed.

    Note that the drifts are only computed the first time they are needed, so
    many Snapshots can be opened at once without needing to wait for all of
    them to compute their drift positions and velocities.

    The drift computation is very simple: it is just a weighted average of all
    particle coordinates/velocities, using the particle masses as the weights.
    If you instead prefer a different computation, feel free to manually set
    the values of `drift_pos` and `drift_vel` yourself. Just be sure to use
    `remove_drift=False` if you call `get_pos()` or `get_vel()` as part of your
    computation.
    """

    def __init__(self, filepath, drift_part_types=True):
        super().__init__(filepath)

        self.drift_pts = self._normalize_part_types(drift_part_types)
        self.drift_pos = None
        self.drift_vel = None

    def _normalize_part_types(self, pts):
        if isinstance(pts, int):
            pts = [pts]
        elif isinstance(pts, str):
            if str.isdigit():
                pts = [int(pts)]
            else:
                raise ValueError(f"don't know what to do with this: {pts}")
        elif pts:
            # open the snapshot to see what PartTypes exist
            with h5py.File(self.filepath, "r") as f:
                keys = list(f.keys())
            pts = [int(k[-1]) for k in keys if k.startswith("PartType")]
        return pts

    def compute_drift_pos(self):
        if not self.drift_pts:
            return np.array([0, 0, 0])

        pos = self.get_pos(self.drift_pts, remove_drift=False)
        mass = self.get_mass(self.drift_pts)
        return np.average(pos, axis=1, weights=mass)

    def compute_drift_vel(self):
        if not self.drift_pts:
            return np.array([0, 0, 0])

        vel = self.get_vel(self.drift_pts, remove_drift=False)
        mass = self.get_mass(self.drift_pts)
        return np.average(vel, axis=1, weights=mass)

    def get_pos(self, part_type, remove_drift=True):
        """Returns the (sorted) Coordinates dataset for `part_type`."""
        pos_raw = self.get_sorted(part_type, "Coordinates")
        if not remove_drift:
            return pos_raw
        if self.drift_pos is None:
            self.drift_pos = self.compute_drift_pos()
        return pos_raw - self.drift_pos

    def get_vel(self, part_type, remove_drift=True):
        """Returns the (sorted) Velocities dataset for `part_type`."""
        vel_raw = self.get_sorted(part_type, "Velocities")
        if not remove_drift:
            return vel_raw
        if self.drift_vel is None:
            self.drift_vel = self.compute_drift_vel()
        return vel_raw - self.drift_vel


class MultiGalaxySnapshot(Snapshot):
    """Extension of the Snapshot class for multi-galaxy sims.

    The MultiGalaxySnapshot allows you to name individual subsets of particles
    within each particle type so that you can query only the subsets of
    particles that correspond to a given galaxy. One can specify these either
    via automatic inferrence, semi-automatic inferrence, or fully manually.

    The data is stored in a dict called `groups`, which is structured as
    follows:
    {
        "group_name": { "type": <int>, "ids": <numpy array, dtype int>, ... }
    }

    If you want to specify this dict manually, pass a dict with this
    information into the `infer_galaxies` argument. Otherwise, pass
    `infer_galaxies=False`, manually set `.groups`, and run `.fill_groups()`
    afterward.

    Otherwise, groups will be inferred automatically under the assumption that
    each group of particles has a unique particle mass. They will be named
    <gal_name>_<part_type>, where the galaxy name is from ['host', 'sat1',
    '...', 'satN']. Galaxies are assigned from largest to smallest total mass
    (per group). For example, in a simulation with two unique DM particle
    masses, two unique stellar disk particle masses (PartType2), and one unique
    stellar bulge particle mass (PartType3), the `p_groups` would look like...
    {
        "host_1": { "type": 1, "ids": <array>, ... },
        "host_2": { "type": 2, "ids": <array>, ... },
        "host_3": { "type": 3, "ids": <array>, ... },
        "sat1_1": { "type": 1, "ids": <array>, ... },
        "sat1_2": { "type": 2, "ids": <array>, ... },
    },
    where `"host_1"` and `"host_2"` correspond to the unique particle masses in
    PartTypes 1 and 2 with the largest _total_ mass. (Note that you can rename
    these after the fact if you want, in case this scheme fails for your use
    case but you prefer using the inferred groups.) To use this scheme, pass
    `infer_galaxies=True` to the constructor.

    There is a third method: semi-automatic inferrence. In this case, pass a
    dictionary to `infer_galaxies` with the following information:
    {
        "group_name": { "type": <int>, "number": <int> },
    }
    This is basically the same as the `groups` dict, except instead of full ID
    lists for each group, you only need to specify the number of particles as
    "number". Then, inferrence will take place as usual, but the results will
    be renamed with any group of the same particle type and number taking your
    specified name. Note that this requires all particle numbers for each
    PartType to be unique.
    """

    def __init__(self, filepath, infer_galaxies=True, drift_part_groups="host*"):
        super().__init__(filepath)

        if infer_galaxies:
            if isinstance(infer_galaxies, dict):
                for group in infer_galaxies.values():
                    if "mask" not in group:
                        break
                else:
                    self.groups = infer_galaxies
                    self.fill_groups()

            self.groups = self._infer_galaxies(infer_galaxies)
        else:
            print("Warning: particle groups not inferred!")

        self.drift_pgs = self._normalize_part_groups(drift_part_groups)
        self.drift_pos = None
        self.drift_vel = None

    def _infer_galaxies(self, infer_galaxies):
        # find all the available particle types
        with h5py.File(self.filepath, "r") as f:
            keys = list(f.keys())
        pts = [int(k[-1]) for k in keys if k.startswith("PartType")]

        def _get_mass_counts(pt):
            # split by particle mass
            mass = self.get_mass(pt)
            unique, counts = np.unique(mass, return_counts=True)
            totals = unique * counts

            # sort from biggest to smallest (_total_ mass)
            sort_idx = np.argsort(totals)[::-1]
            unique = unique[sort_idx]
            counts = counts[sort_idx]
            totals = totals[sort_idx]

            return mass, unique, counts, totals

        # *** Identify groups for each particle type ***
        groups = {}
        for pt in pts:
            # split by particle mass
            mass, unique, counts, totals = _get_mass_counts(pt)

            # these are all sorted by particle ID
            ids = self.get_ids(pt)
            pos = self.get_pos(pt, remove_drift=False)
            vel = self.get_vel(pt, remove_drift=False)

            # figure out how to unsort back to raw particle ID order
            raw_ids = self[f"PartType{pt}/ParticleIDs"]
            unsort_idx = np.argsort(np.argsort(raw_ids))

            for i, (u, c, t) in enumerate(zip(unique, counts, totals)):
                mask = mass == u
                com = np.average(pos[mask], axis=1, weights=mass[mask])
                cov = np.average(vel[mask], axis=1, weights=mass[mask])

                name = ("host" if i == 0 else f"sat{i}") + f"_{pt}"

                groups[name] = {
                    "type": pt,
                    "ids": ids[mask],
                    "mask": mask,
                    "mask_raw": mask[unsort_idx],
                    "number": c,
                    "total_mass": t,
                    "com": com,
                    "cov": cov,
                }

        if isinstance(infer_galaxies, dict):
            # rename in accordance with provided information
            for n, group in groups.items():
                for name, info in infer_galaxies.items():
                    if (
                        info["type"] == group["type"]
                        and info["number"] == group["number"]
                    ):
                        groups[name] = group
                        del groups[n]
                        break

        return groups

    def fill_groups(self):
        for name, group in self.groups.items():
            raw_ids = self[f"PartType{group['type']}"]
            sort_idx = np.argsort(raw_ids)
            unsort_idx = np.argsort(sort_idx)

            # make mask from known IDs
            ids = raw_ids[sort_idx]
            indices = np.searchsorted(ids, group["ids"])
            mask = np.zeros(len(ids), dtype=bool)
            mask[indices] = True

            mass = self.get_mass(group["type"])[mask]
            count = len(mass)
            total = mass.sum()

            pos = self.get_pos(group["type"], remove_drift=False)[group["mask"]]
            com = np.average(pos, axis=1, weights=mass)

            vel = self.get_vel(group["type"], remove_drift=False)[group["mask"]]
            cov = np.average(vel, axis=1, weights=mass)

            self.groups[name]["mask"] = mask
            self.groups[name]["mask_raw"] = mask[unsort_idx]
            self.groups[name]["number"] = count
            self.groups[name]["total_mass"] = total
            self.groups[name]["com"] = com
            self.groups[name]["cov"] = cov

    def _normalize_part_groups(self, pgs):
        if not pgs:
            return False

        if isinstance(pgs, int):
            raise ValueError("need to specify particle group names, not particle types")

        if isinstance(pgs, str):
            # check if it is a wildcard
            if pgs.endswith("*"):
                pattern = pgs[:-1]
                pgs = [n for n in self.groups.keys() if n.startswith(pattern)]
                if len(pgs) == 0:
                    print("warning: no pgs found matching pattern %s" % pattern + "*")

            else:
                pgs = [pgs]

        else:
            for pg in pgs:
                if not isinstance(pg, str):
                    raise ValueError("only strings allowed as particle group names")

                if pg not in list(self.groups.keys()):
                    raise ValueError("unrecognized particle group: %s" % pg)

        if len(pgs) == 0:
            return False

        return pgs

    def compute_drift_pos(self):
        if not self.drift_pgs:
            return np.array([0, 0, 0])

        pos = self.get_pos(self.drift_pgs, remove_drift=False)
        mass = self.get_mass(self.drift_pgs)
        return np.average(pos, axis=1, weights=mass)

    def compute_drift_vel(self):
        if not self.drift_pgs:
            return np.array([0, 0, 0])

        vel = self.get_vel(self.drift_pgs, remove_drift=False)
        mass = self.get_mass(self.drift_pgs)
        return np.average(vel, axis=1, weights=mass)

    def _type_and_mask(self, pg):
        """Given a group name or particle type, return the particle type and mask."""
        try:
            g = self.groups[pg]
        except KeyError:
            # not a particle group, assume it's a particle type:
            if isinstance(pg, str) and pg.isdigit():
                pg = int(pg)
            if isinstance(pg, int):
                return pg, slice(None)
            else:
                raise ValueError(f"don't know what to do with this: {pg}")

        return g["type"], g["mask_raw"]

    def _many_or_single(self, single_func, pgs, *args, **kwargs):
        if not isinstance(pgs, (tuple, list)):
            pgs = [pgs]
        if len(pgs) == 1:
            return single_func(pgs[0], *args, **kwargs)
        return np.concatenate([single_func(pg, *args, **kwargs) for pg in pgs], axis=0)

    def get(self, part_group, key):
        """Gets the chunk of dataset 'key' associated with 'part_group'."""

        def _get(pg, key):
            pt, mask = self._type_and_mask(pg)
            return self[f"PartType{pt}/{key}", mask]

        return self._many_or_single(_get, part_group, key)

    def get_sorted(self, part_group, key):
        """Gets the chunk of dataset 'key' associated with 'part_group'."""

        def _get_sorted(pg, key):
            pt, mask = self._type_and_mask(pg)
            ids = self[f"PartType{pt}/ParticleIDs", mask]
            val = self[f"PartType{pt}/{key}", mask]
            sort_idx = np.argsort(ids)
            return val[sort_idx]

        return self._many_or_single(_get_sorted, part_group, key)

    def get_pos(self, part_group, remove_drift=True):
        """Returns the (sorted) Coordinates dataset for `part_group`."""
        pos_raw = self.get_sorted(part_group, "Coordinates")
        if not remove_drift:
            return pos_raw
        if self.drift_pos is None:
            self.drift_pos = self.compute_drift_pos()
        return pos_raw - self.drift_pos

    def get_vel(self, part_group, remove_drift=True):
        """Returns the (sorted) Velocities dataset for `part_group`."""
        vel_raw = self.get_sorted(part_group, "Velocities")
        if not remove_drift:
            return vel_raw
        if self.drift_vel is None:
            self.drift_vel = self.compute_drift_vel()
        return vel_raw - self.drift_vel
