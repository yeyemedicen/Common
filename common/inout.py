''' input/output module '''


def read_HDF5_data(comm, h5file, fun, name):
    ''' Read checkpoint data from an XDMF file into a dolfinx Function.

    Args:
        comm            MPI communicator (e.g. mesh.comm)
        h5file (str)    Path to checkpoint file. If the extension is `.h5`
                        the equivalent `.xdmf` file is used instead
                        (legacy FEniCS .h5 checkpoint files are not directly
                        readable by DOLFINx).
        fun             dolfinx Function to read into
        name (str)      Name of the dataset (used as the function name)

    Returns:
        time            Timestamp stored in the file, or 0 if none
    '''
    from dolfinx.io import XDMFFile

    xdmf_file = h5file[:-3] + '.xdmf' if h5file.endswith('.h5') else h5file
    time = 0.0
    with XDMFFile(comm, xdmf_file, 'r') as xf:
        xf.read_function(fun, name)
    return time


def old_read_HDF5_data(mpi_comm, h5file, fun, name):
    ''' Read checkpoint data from a HDF5 file into a dolfin function.

    Args:
        h5file (str)    HDF5 File to be read from
        mpi_comm        MPI comm, e.g. mesh.mpi_comm()
        fun             Dolfin function
        name (str)      name of the hdf5 dataset

    Returns:
        time            timestamp if solution, 0 if none given
    '''
    from dolfin import HDF5File

    with HDF5File(mpi_comm, h5file, 'r') as hdf:
        hdf.read(fun, name)
        time = 0
        if 'timestamp' in hdf.attributes(name + '/vector_0'):
            time = hdf.attributes(name + '/vector_0')['timestamp']

    return time


def write_HDF5_data(comm, h5file, fun, name, t=0.):
    ''' Write checkpoint data from a dolfinx Function into an XDMF file.

    The output is written as a pair of files: `<stem>.xdmf` (XML header)
    and `<stem>.h5` (binary data), which together form the DOLFINx XDMF
    checkpoint format.

    Args:
        comm            MPI communicator (e.g. mesh.comm)
        h5file (str)    Output path. The `.h5` extension is replaced with
                        `.xdmf` so that both the XDMF header and the HDF5
                        data backend are written alongside each other.
        fun             dolfinx Function to write
        name (str)      Name of the dataset
        t (float)       Timestamp
    '''
    from dolfinx.io import XDMFFile

    xdmf_file = h5file[:-3] + '.xdmf' if h5file.endswith('.h5') else h5file
    with XDMFFile(comm, xdmf_file, 'w') as xf:
        xf.write_function(fun, float(t))


def write_HDF5_data_old(mpi_comm, h5file, fun, name, t=0.):
    ''' Write checkpoint data from a dolfin function into a HDF5 file for
    reuse.

    Args:
        h5file (str)    HDF5 File to be read from
        mpi_comm        MPI comm, e.g. mesh.mpi_comm()
        fun             Dolfin function
        name (str)      name of the hdf5 dataset
    '''
    from dolfin import HDF5File

    with HDF5File(mpi_comm, h5file, 'w') as hdf:
        hdf.write(fun, name, float(t))

def read_mesh(mesh_file):
    ''' Read mesh and boundary/subdomain tags for DOLFINx.

    Supported format: XDMF (.xdmf).

    The function expects either:
    * A single XDMF file containing the mesh **and** meshtags stored under
      the names ``"subdomains"`` and ``"boundaries"``, or
    * A mesh XDMF file alongside ``<stem>_subdomains.xdmf`` and
      ``<stem>_boundaries.xdmf`` companion files.

    Legacy FEniCS `.h5` and `.xml` mesh files are **not** supported.
    Convert them first::

        python -m meshio convert mesh.h5 mesh.xdmf

    Args:
        mesh_file (str)   Path to the XDMF mesh file

    Returns:
        mesh              dolfinx.mesh.Mesh
        subdomains        dolfinx.mesh.MeshTags for cell markers (or None)
        boundaries        dolfinx.mesh.MeshTags for facet markers (or None)
    '''
    from mpi4py import MPI
    from dolfinx.io import XDMFFile

    tmp = mesh_file.split('.')
    file_type = tmp[-1]
    mesh_stem = '.'.join(tmp[:-1])

    if file_type in ('h5', 'xml'):
        raise NotImplementedError(
            'Legacy FEniCS {} mesh format is not supported by DOLFINx. '
            'Convert to XDMF first, e.g.:\n'
            '  python -m meshio convert {} {}.xdmf'.format(
                file_type, mesh_file, mesh_stem))

    if file_type != 'xdmf':
        raise Exception(
            'Mesh format ".{}" not recognised. Use XDMF (.xdmf).'.format(
                file_type))

    # ── Read mesh ─────────────────────────────────────────────────────────────
    mesh = None
    for _mesh_name in ('Grid', 'mesh', 'Mesh'):
        try:
            with XDMFFile(MPI.COMM_WORLD, mesh_file, 'r') as xf:
                mesh = xf.read_mesh(name=_mesh_name)
            break
        except Exception:
            continue
    if mesh is None:
        raise RuntimeError(
            'Could not read mesh from {}. Tried grid names: Grid, mesh, Mesh.'
            .format(mesh_file))

    # Connectivity needed before reading facet meshtags
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # ── Try reading tags from the same XDMF file first ────────────────────────
    # Try multiple common naming conventions
    _cell_names = ('subdomains', 'cells', 'Grid')
    _facet_names = ('boundaries', 'facets', 'Grid')

    subdomains = None
    for _name in _cell_names:
        subdomains = _try_read_meshtags(MPI.COMM_WORLD, mesh_file, mesh,
                                        _name, mesh.topology.dim)
        if subdomains is not None:
            break

    boundaries = None
    for _name in _facet_names:
        boundaries = _try_read_meshtags(MPI.COMM_WORLD, mesh_file, mesh,
                                        _name, mesh.topology.dim - 1)
        if boundaries is not None:
            break

    # ── Fall back to companion files if not found ──────────────────────────────
    if subdomains is None:
        companion = mesh_stem + '_subdomains.xdmf'
        for _name in ('Grid', 'subdomains', 'cells'):
            subdomains = _try_read_meshtags(MPI.COMM_WORLD, companion, mesh,
                                            _name, mesh.topology.dim)
            if subdomains is not None:
                break

    if boundaries is None:
        companion = mesh_stem + '_boundaries.xdmf'
        for _name in ('Grid', 'boundaries', 'facets'):
            boundaries = _try_read_meshtags(MPI.COMM_WORLD, companion, mesh,
                                            _name, mesh.topology.dim - 1)
            if boundaries is not None:
                break

    if boundaries is None and MPI.COMM_WORLD.rank == 0:
        print('Warning: no boundary tags found for mesh {}'.format(mesh_file))

    return mesh, subdomains, boundaries


def read_mesh_old(mesh_file):
    ''' Read HDF5 or DOLFIN XML mesh.

    Args:
        mesh_file       path to mesh file

    Returns:
        mesh            Mesh
        sd              subdomains
        bnd             boundaries
    '''
    from dolfin import Mesh, MeshFunction, HDF5File, XDMFFile
    # pth = '/'.join(mesh_file.split('/')[0:-1])
    tmp = mesh_file.split('.')  # [-1].split('.')
    file_type = tmp[-1]
    mesh_pref = '.'.join(tmp[0:-1])

    if file_type == 'xml':
        mesh = Mesh(mesh_file)
        try:
            subdomains = MeshFunction('size_t', mesh,
                                      mesh_pref + '_physical_region.xml')
        except RuntimeError:
            subdomains = MeshFunction('int', mesh,
                                      mesh_pref + '_physical_region.xml')
        except FileNotFoundError:
            subdomains = MeshFunction('size_t', mesh,
                                      mesh.topology().dim())

        try:
            boundaries = MeshFunction('size_t', mesh,
                                      # mesh.topology().dim() - 1,
                                      mesh_pref + '_facet_region.xml')
        except RuntimeError:
            boundaries = MeshFunction('int', mesh,
                                      # mesh.topology().dim() - 1,
                                      mesh_pref + '_facet_region.xml')
        except FileNotFoundError:
            if mesh.mpi_comm().Get_rank() == 0:
                print('no boundary file found ({})'.format(
                    mesh_pref+'_facet_region.xml'))
            boundaries = MeshFunction('size_t', mesh,
                                      mesh.topology().dim() - 1)

    elif file_type == 'h5':
        mesh = Mesh()

        with HDF5File(mesh.mpi_comm(), mesh_file, 'r') as hdf:
            hdf.read(mesh, '/mesh', False)
            subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())
            boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()
                                      - 1)

            if hdf.has_dataset('subdomains'):
                hdf.read(subdomains, '/subdomains')

            if hdf.has_dataset('boundaries'):
                hdf.read(boundaries, '/boundaries')
            else:
                if mesh.mpi_comm().Get_rank() == 0:
                    print('no <boundaries> datasets found in file {}'.format(
                        mesh_file))

    elif file_type == 'xdmf':
        import os
        import h5py as _h5py
        from dolfin import MPI as _dMPI, MeshValueCollection

        _comm      = _dMPI.comm_world
        _comm_self = _dMPI.comm_self
        _rank      = _comm.rank
        _parallel  = (_comm.size > 1)

        bnd_companion = mesh_pref + '_boundaries.xdmf'
        _bnd_h5       = mesh_pref + '.h5'

        # Detect whether boundary entities are cell-level (same dim as mesh
        # cells, e.g. triangle patches on a surface mesh) or facet-level (one
        # dim lower, e.g. edge labels on a 2-D mesh).
        # Cell labels → read into subdomains / use dx(tag).
        # Facet labels → read into boundaries / use ds(tag).
        _bnd_is_cell = False
        if os.path.isfile(_bnd_h5):
            with _h5py.File(_bnd_h5, 'r') as _hf:
                if 'boundaries/topology' in _hf and 'mesh/topology' in _hf:
                    _bnd_is_cell = (
                        _hf['boundaries/topology'].shape[1]
                        == _hf['mesh/topology'].shape[1]
                    )

        if not _parallel:
            # ── Serial path: read XDMF directly ───────────────────────────────
            mesh = Mesh()
            with XDMFFile(_comm, mesh_file) as xf:
                xf.read(mesh)

            dim = mesh.topology().dim()
            subdomains = MeshFunction('size_t', mesh, dim, 0)
            boundaries = MeshFunction('size_t', mesh, dim - 1, 0)

            if os.path.isfile(bnd_companion):
                try:
                    if _bnd_is_cell:
                        # Triangle patches → cell labels → subdomains
                        mvc = MeshValueCollection('size_t', mesh, dim)
                        with XDMFFile(_comm, bnd_companion) as xf:
                            xf.read(mvc, 'boundaries')
                        for (cell_idx, _li), val in mvc.values().items():
                            subdomains[cell_idx] = val
                    else:
                        # Edge labels → facet labels → boundaries
                        mvc = MeshValueCollection('size_t', mesh, dim - 1)
                        with XDMFFile(_comm, bnd_companion) as xf:
                            xf.read(mvc, 'boundaries')
                        mesh.init(dim, dim - 1)
                        c2f = mesh.topology()(dim, dim - 1)
                        for (cell_idx, local_idx), val in mvc.values().items():
                            boundaries[c2f(cell_idx)[local_idx]] = val
                except Exception as e:
                    print('Warning: could not read boundaries from {}: {}'.format(
                        bnd_companion, e))
            else:
                print('Warning: no boundary file found ({})'.format(bnd_companion))

        else:
            # ── Parallel path ─────────────────────────────────────────────────
            # dolfin 2019.x XDMFFile.read() calls a global MPI collective
            # internally even when passed comm_self, so it cannot be called on
            # only one rank of a multi-rank job — the other ranks never
            # participate and the collective deadlocks.
            #
            # Fix: rank 0 launches a fresh single-process subprocess (no MPI
            # context) that reads the XDMF and writes a proper dolfin-format
            # HDF5 temp file.  HDF5File parallel reading (used by all ranks
            # below) is unaffected by the MPI ABI mismatch and works correctly.
            import subprocess as _subp
            import sys as _sys
            import textwrap as _tw

            tmp_h5 = mesh_pref + '_parallel_tmp.h5'

            if _rank == 0:
                _script = _tw.dedent("""\
                    from dolfin import (Mesh, XDMFFile, MeshFunction,
                                        MeshValueCollection, HDF5File, MPI as dMPI)
                    import os, sys
                    comm_self    = dMPI.comm_self
                    mesh_file    = sys.argv[1]
                    bnd_companion= sys.argv[2]
                    tmp_h5       = sys.argv[3]
                    bnd_is_cell  = sys.argv[4] == '1'

                    m = Mesh()
                    with XDMFFile(comm_self, mesh_file) as xf:
                        xf.read(m)
                    dim = m.topology().dim()
                    sub = MeshFunction('size_t', m, dim, 0)
                    bnd = MeshFunction('size_t', m, dim - 1, 0)
                    if os.path.isfile(bnd_companion):
                        try:
                            if bnd_is_cell:
                                mvc = MeshValueCollection('size_t', m, dim)
                                with XDMFFile(comm_self, bnd_companion) as xf:
                                    xf.read(mvc, 'boundaries')
                                for (ci, _li), val in mvc.values().items():
                                    sub[ci] = val
                            else:
                                mvc = MeshValueCollection('size_t', m, dim - 1)
                                with XDMFFile(comm_self, bnd_companion) as xf:
                                    xf.read(mvc, 'boundaries')
                                m.init(dim, dim - 1)
                                c2f = m.topology()(dim, dim - 1)
                                for (ci, li), val in mvc.values().items():
                                    bnd[c2f(ci)[li]] = val
                        except Exception as e:
                            print('Warning: could not read boundaries:', e)
                    else:
                        print('Warning: no boundary file found:', bnd_companion)
                    with HDF5File(comm_self, tmp_h5, 'w') as hdf:
                        hdf.write(m, '/mesh')
                        hdf.write(sub, '/subdomains')
                        hdf.write(bnd, '/boundaries')
                """)
                # Strip MPI environment variables so the subprocess starts as a
                # plain serial Python process — without this, dolfin/mpi4py
                # detects PMI_FD and re-joins the MPI job, causing the same
                # XDMFFile collective deadlock inside the subprocess.
                import os as _os
                _env = {k: v for k, v in _os.environ.items()
                        if not any(k.startswith(_p) for _p in (
                            'PMI_', 'MPI_', 'OMPI_', 'MPICH_', 'PMIX_',
                            'HYDRA_', 'PRTE_', 'PRRTE_', 'ORTED_',
                            'ORTE_', 'MPIEXEC_', 'I_MPI_'))}
                _result = _subp.run(
                    [_sys.executable, '-c', _script,
                     mesh_file, bnd_companion, tmp_h5,
                     '1' if _bnd_is_cell else '0'],
                    capture_output=True, text=True, env=_env)
                if _result.returncode != 0:
                    raise RuntimeError(
                        'XDMF→HDF5 conversion subprocess failed:\n' +
                        _result.stderr)
                if _result.stderr:
                    print(_result.stderr, end='')

            _dMPI.barrier(_comm)

            mesh = Mesh()
            with HDF5File(_comm, tmp_h5, 'r') as hdf:
                hdf.read(mesh, '/mesh', False)
                dim = mesh.topology().dim()
                subdomains = MeshFunction('size_t', mesh, dim, 0)
                boundaries = MeshFunction('size_t', mesh, dim - 1, 0)
                if hdf.has_dataset('subdomains'):
                    hdf.read(subdomains, '/subdomains')
                if hdf.has_dataset('boundaries'):
                    hdf.read(boundaries, '/boundaries')

            if _rank == 0:
                os.remove(tmp_h5)

    else:
        raise Exception('Mesh format not recognized. Try XDMF or HDF5 (or XML,'
                        ' deprecated)')

    return mesh, subdomains, boundaries


def _try_read_meshtags(comm, xdmf_file, mesh, name, dim):
    ''' Attempt to read meshtags from an XDMF file; return None on failure. '''
    import os
    from dolfinx.io import XDMFFile

    if not os.path.isfile(xdmf_file):
        return None
    try:
        with XDMFFile(comm, xdmf_file, 'r') as xf:
            tags = xf.read_meshtags(mesh, name=name)
        return tags
    except Exception:
        return None


def read_parameters(infile):
    ''' Read in parameters yaml file.

    Args:
        infile      path to yaml file

    Return:
        prms        parameters dictionary
    '''
    import ruamel.yaml as yaml
    with open(infile, 'r+') as f:
        yaa = yaml.YAML(typ='rt')
        prms = yaa.load(f)
    return prms


def dump_parameters(prms):
    ''' Wrapper for yaml.dump (e.g., for logging.debug()) '''
    import ruamel.yaml as yaml
    import sys
    yaa = yaml.YAML(typ='unsafe', pure=True)
    return yaa.dump(prms, sys.stdout)


def print_parameters(prms):
    ''' Print parameter dictionary in human readable form.

    Args:
        prms        parameters dictionary
    '''
    import ruamel.yaml as yaml
    print(yaml.dump(prms))
