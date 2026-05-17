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
    from dolfin import Mesh, MeshFunction, HDF5File, XDMFFile, edges as _edges, vertices as _verts
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

        mesh = Mesh()
        with XDMFFile(mesh_file) as xf:
            xf.read(mesh)

        # Build ALL connectivity tables (vertex↔edge needed for tag lookup).
        mesh.init()

        subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)

        # ── Read boundary tags from HDF5 directly ─────────────────────────────
        # dolfin's XDMFFile.read(MeshFunction) is unreliable with externally
        # written XDMF (state corruption, Precision mismatches, …).  Instead
        # we read the topology (edge vertex-pairs) and values straight from
        # the companion .h5 file, then populate the MeshFunction ourselves
        # using dolfin's vertex-to-edge connectivity table.
        h5_file = mesh_pref + '.h5'
        if os.path.isfile(h5_file):
            try:
                with _h5py.File(h5_file, 'r') as hf:
                    bnd_topo = hf['boundaries/topology'][()]   # (N_bnd, 2)
                    bnd_vals = hf['boundaries/values'][()]     # (N_bnd,)

                # Build vertex → edge-index map from dolfin's own iterators.
                # mesh.topology()(0,1) is unreliable across dolfin versions;
                # iterating edges() is always correct.
                mesh.init(1)   # ensure edge entities are created
                v2e = {}
                for e in _edges(mesh):
                    for v in _verts(e):
                        v2e.setdefault(v.index(), []).append(e.index())

                n_set = 0
                for (v0, v1), tag in zip(bnd_topo, bnd_vals):
                    shared = set(v2e.get(int(v0), [])) & set(v2e.get(int(v1), []))
                    if shared:
                        boundaries[shared.pop()] = int(tag)
                        n_set += 1

                import numpy as _np
                unique_tags = _np.unique(boundaries.array())
                print('Boundary tags loaded: {} / {} edges tagged  |  tags: {}'.format(
                    n_set, len(bnd_vals), unique_tags.tolist()))
            except Exception as e:
                print('Warning: could not read boundary tags from {}: {}'.format(
                    h5_file, e))

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
