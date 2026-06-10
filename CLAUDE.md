# Common utilities — notes for Claude

## Project layout

- `common/inout.py` — mesh I/O for both legacy FEniCS (dolfin) and DOLFINx.
  - `read_mesh_old` — legacy FEniCS (conda env). Handles `.xml`, `.h5`, `.xdmf`.
  - `read_mesh` — DOLFINx.
- Meshes come from `/Users/yeye/MeasureIt/utils.py` (`_write_xdmf`), which produces
  three files per mesh: `mesh.xdmf`, `mesh.h5`, `mesh_boundaries.xdmf`.

## Known bugs / hard-won fixes

### `read_mesh_old` XDMF branch — MPI and default-value bugs (fixed)

**Symptom (MPI):** running with `mpirun -n N` produced wrong or empty boundary tags
on non-root processes.

**Root cause:** the original implementation read boundary topology from the `.h5` file
directly via `h5py` and matched entities using Python vertex-to-entity loops. This is
not MPI-aware: each process read the full file independently, and after mesh
partitioning the local entity indices no longer correspond to global vertex indices.

**Fix:** replaced h5py reading with `MeshValueCollection` + `XDMFFile(mesh.mpi_comm(),
bnd_companion)`. Dolfin's XDMF reader handles MPI partitioning internally.

---

**Symptom (wrong tags):** `np.unique(boundaries.array())` returned `[1, 2, 3, 4]` with
no zeros — interior facets were not zero, breaking `DirichletBC` on tag 1.

**Root cause:** `MeshFunction('size_t', mesh, mvc)` does **not** guarantee zero for
unassigned entries in all dolfin builds. Interior facets (not present in the MVC) got
garbage / non-zero values instead of 0.

**Fix:** keep `boundaries = MeshFunction('size_t', mesh, dim - 1, 0)` (zero-initialized)
and assign from the MVC manually using topology connectivity:

```python
mesh.init(dim, dim - 1)
c2f = mesh.topology()(dim, dim - 1)
for (cell_idx, local_idx), val in mvc.values().items():
    boundaries[c2f(cell_idx)[local_idx]] = val
```

This is MPI-safe: `mvc.values()` on each process contains only local entries, and
`c2f(cell_idx)[local_idx]` gives the local facet index. Interior facets are never
touched and remain 0.

**Do not** replace this with `MeshFunction('size_t', mesh, mvc)` — it looks simpler
but has undefined behavior for unassigned entries.

---

### `XDMFFile` parallel reading is BROKEN on this Mac conda install — subprocess workaround

**Symptom:** `mpirun -n 2 python script.py mesh.xdmf` hangs forever — no output, no
error, no timeout. The hang occurs inside `read_mesh_old` before any user code prints.

**Root cause — two compounding problems:**

1. **MPICH ABI mismatch.** Dolfin links `libmpi.12.dylib` (OpenMPI ABI) but the conda
   env has MPICH 4.3.0 (`MPI.Session` struct size 32 vs 40). `XDMFFile` parallel
   reading goes through the broken mpi4py conversion and silently drops ~40% of mesh
   vertices.  `HDF5File` uses a different code path not affected by this.

2. **`XDMFFile.read()` is not truly serial even with `comm_self`.** Even when
   `XDMFFile` is constructed with `dMPI.comm_self`, the `.read(mesh)` call makes a
   global MPI collective internally. If only rank 0 calls it (rank 1 is waiting at
   a barrier), rank 1 never participates in the collective → deadlock.

**Attempted fix that does NOT work:** calling `XDMFFile(comm_self, …).read(m)` inside
`if rank == 0:`. Confirmed by bisection: the hang occurs exactly at `xf.read(m)`.

**Actual fix (in `read_mesh_old` XDMF parallel branch):** rank 0 launches a
**subprocess** with all MPI environment variables stripped, so it starts as a true
single-process Python with no MPI context. That subprocess reads the XDMF and writes
a dolfin-format temp HDF5. Then all ranks read the temp HDF5 in parallel with
`HDF5File` (which works). Rank 0 deletes the temp file.

**Why the subprocess must strip MPI env vars:** `mpirun` sets `PMI_FD`, `PMI_RANK`,
`PMI_SIZE`, `MPI_LOCALRANKID`, etc. in the process environment. A subprocess inherits
these. When dolfin/mpi4py initialises inside the subprocess, it detects `PMI_FD` and
re-joins the same MPI job — putting us back to the deadlock. The fix strips all `PMI_`,
`MPI_`, `OMPI_`, `MPICH_`, `PMIX_`, `HYDRA_`, `PRTE_`, `ORTE_`, `MPIEXEC_`, `I_MPI_`
prefixed variables from the environment passed to `subprocess.run`.

The temp file is named `<stem>_parallel_tmp.h5` (next to the XDMF).

**Do NOT** try `XDMFFile(comm_self, …).read(m)` inside `if rank == 0:` — it deadlocks.
**Do NOT** strip only some PMI vars — `PMI_FD` alone is enough to re-join the job.

**Long-term fix:** reinstall mpi4py against the same MPICH version as dolfin:
```
conda install --force-reinstall mpi4py
```
(only works if conda has mpi4py built against the same MPICH ABI as dolfin).

### `_boundaries.xdmf` format and cell-vs-facet label routing (fixed)

**Symptom:** `assemble(Constant(1.0) * ds(tag))` always returns 0.0 for meshes
written by MeasureIt with `is_3d=True` (e.g. `kink3dl.xdmf`), even though
`np.unique(boundaries.array())` showed non-zero tags.

**Root cause:** MeasureIt's `_write_xdmf` writes **different topology** depending on
the mesh type:

| `is_3d` | boundary entity | `TopologyType` | correct target |
|---------|----------------|----------------|----------------|
| `False` | edges          | `Interval` (npe=2) | `boundaries` MeshFunction / `ds(tag)` |
| `True`  | triangles      | `Triangle` (npe=3) | `subdomains` MeshFunction / `dx(tag)` |

For `is_3d=True`, the boundary file stores **surface patches** (one label per
triangle), not edge labels.  The old code always read into `boundaries` (dim-1
facets), so for a closed 3D surface mesh — where every edge is interior — `ds`
was identically zero.

**Fix (in `read_mesh_old` XDMF branch):** detect the entity kind by comparing
`boundaries/topology.shape[1]` vs `mesh/topology.shape[1]` in the `.h5` file:
- equal → cell labels (same npe) → read as `MeshValueCollection` at `dim` → assign to `subdomains`
- smaller → facet labels → read as `MeshValueCollection` at `dim-1` → assign to `boundaries`

This applies to both the serial and parallel paths.

**Usage after the fix:**
- 3D surface mesh patches: use `dx = Measure('dx', domain=mesh, subdomain_data=subdomains)` and `dx(tag)`
- 2D mesh edges: use `ds = Measure('ds', domain=mesh, subdomain_data=boundaries)` and `ds(tag)`

**Note on `mpirun` hangs:** PETSc collective operations (`.norm()`, `dot()`, any
`VecNorm`/`VecDot` internally) must be called by **all** MPI ranks simultaneously.
Calling them inside `if rank == 0:` causes the other rank to wait in
`MPI_Allreduce` while rank 0 calls the collective — deadlock. Always compute the
value on all ranks first, then gate only the print statement.

---

### Parallel programming patterns — collective operation deadlocks

These bugs are easy to introduce and hard to diagnose (the process just hangs forever
with no error message). Confirmed cases in this codebase:

**Pattern 1 — rank-local tag arrays used for collective loops.**
`MeshFunction.array()` (legacy) and `MeshTags.values` (DOLFINx) are rank-local: each
process only holds the entities in its partition. Different ranks may see different
subsets of tags. If you loop `for tag in np.unique(local_values): assemble(…)`, rank 0
may loop 4 times and rank 1 only 2 times — `assemble` is collective, so after rank 1's
2nd call both ranks are out of sync → hang on rank 0's 3rd call.

**Fix:** allgather before the loop. Reusable pattern:
```python
# legacy dolfin
def global_tags(mf, comm):
    local = np.unique(mf.array()).astype(np.int64)
    return np.unique(np.concatenate(comm.allgather(local)))

# DOLFINx
def global_tags(mt, comm):
    local = np.unique(mt.values).astype(np.int32)
    return np.unique(np.concatenate(comm.allgather(local)))
```

**Pattern 2 — DOLFINx `assemble_scalar` is rank-local.**
Unlike legacy dolfin (which returns the global reduced scalar), DOLFINx
`fem.assemble_scalar` returns only this rank's local contribution. You must
`comm.allreduce(local, op=MPI.SUM)` explicitly to get the global integral.
```python
local = fem.assemble_scalar(fem.form(1 * ds(tag)))
area  = comm.allreduce(local, op=MPI.SUM)
```

**Pattern 3 — DOLFINx `LinearProblem` (v0.10.0) requires `petsc_options_prefix`.**
If omitted, `LinearProblem.__init__` raises `TypeError` on rank 0 before any MPI
operations; rank 1 then hangs waiting for a collective that never comes.
Always pass `petsc_options_prefix='some_prefix_'` as a keyword argument.

---

### DOLFINx `read_mesh` — `_boundaries.h5` companion file required

**Symptom:** `read_mesh` prints "no boundary tags found" for XDMF meshes written by
MeasureIt, even though `_boundaries.xdmf` exists alongside the mesh.

**Root cause:** DOLFINx's `XDMFFile` ignores the filename in XDMF DataItem text and
always opens `<xdmf_stem>.h5`. So `kink3dl_boundaries.xdmf` needs a companion
`kink3dl_boundaries.h5` — it cannot share `kink3dl.h5`.

**Fix:** `_write_xdmf` in MeasureIt now writes a `<stem>_boundaries.h5` alongside
`<stem>_boundaries.xdmf`. The internal HDF5 paths (`/mesh/geometry`,
`/boundaries/topology`, `/boundaries/values`) match what the DataItems reference.

**If you have an old labeled mesh without `_boundaries.h5`**, create it manually:
```python
import h5py, numpy as np
src = 'mesh.h5';  dst = 'mesh_boundaries.h5'
with h5py.File(src, 'r') as sf, h5py.File(dst, 'w') as df:
    df.create_group('mesh').create_dataset('geometry', data=sf['mesh/geometry'][:])
    bg = df.create_group('boundaries')
    bg.create_dataset('topology', data=sf['boundaries/topology'][:])
    bg.create_dataset('values',   data=sf['boundaries/values'][:])
```
