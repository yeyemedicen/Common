''' input/output module '''


def read_HDF5_data(mpi_comm, h5file, fun, name):
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


def write_HDF5_data(mpi_comm, h5file, fun, name, t=0.):
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

        mesh = Mesh()

        with XDMFFile(mesh_file) as xf:
            xf.read(mesh)
            subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
            boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()
                                      - 1, 0)

            xf.read(subdomains)
            xf.read(boundaries)

    else:
        raise Exception('Mesh format not recognized. Try XDMF or HDF5 (or XML,'
                        ' deprecated)')


# NOTE http://fenicsproject.org/qa/5337/importing-marked-mesh-for-parallel-use
    # see file xml2xdmf.py
    return mesh, subdomains, boundaries


def read_parameters(infile):
    ''' Read in parameters yaml file.

    Args:
        infile      path to yaml file

    Return:
        prms        parameters dictionary
    '''
    import ruamel.yaml as yaml
    with open(infile, 'r+') as f:
        prms = yaml.load(f, Loader=yaml.Loader)
    return prms


def dump_parameters(prms):
    ''' Wrapper for yaml.dump (e.g., for logging.debug()) '''
    import ruamel.yaml as yaml
    return yaml.dump(prms)


def print_parameters(prms):
    ''' Print parameter dictionary in human readable form.

    Args:
        prms        parameters dictionary
    '''
    import ruamel.yaml as yaml
    print(yaml.dump(prms))
