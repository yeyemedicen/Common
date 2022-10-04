'''
Convert XML mesh to HDF5 using DOLFIN
'''
from dolfin import *
from pathlib import Path


def convert(infile):
    ''' Convert XML to HDF5, including boundary and subdomain tags.

    Args:
        infile      XML mesh file to be converted
    '''
    fin = Path(infile)

    try:
        fin.resolve()
    except FileNotFoundError:
        raise

    if not fin.suffix == '.xml':
        raise Exception('Expected mesh file with .xml extension, got {}'.
                        format(fin.suffix))

    # read mesh
    print('reading DOLFIN mesh {}'.format(fin))
    mesh = Mesh(str(fin))

    # write mesh
    fout = fin.parent / (fin.stem + '.h5')
    print('writing HDF5 mesh {}'.format(fout))
    hdf5 = HDF5File(mesh.mpi_comm(), str(fout), 'w')
    hdf5.write(mesh, '/mesh')

    # if files exist, write subdomain and boundary information
    subdomains = None
    boundaries = None

    fin_subdomains = fin.parent / (fin.stem + '_physical_region.xml')
    if fin_subdomains.exists():

        # size_t for dolfin-convert, int for meshio-convert
        try:
            subdomains = MeshFunction('size_t', mesh, str(fin_subdomains))
        except RuntimeError:
            subdomains = MeshFunction('int', mesh, str(fin_subdomains))

        print('writing subdomain tags')
        hdf5.write(subdomains, '/subdomains')

    else:
        print('no subdomain tags found')

    fin_boundaries = fin.parent / (fin.stem + '_facet_region.xml')
    if fin_boundaries.exists():
        fin_boundaries.resolve()

        # size_t for dolfin-convert, int for meshio-convert
        try:
            boundaries = MeshFunction('size_t', mesh, str(fin_boundaries))
        except RuntimeError:
            boundaries = MeshFunction('int', mesh, str(fin_boundaries))

        print('writing boundary tags')
        hdf5.write(boundaries, '/boundaries')

    else:
        print('no boundary tags found')


def _parse_options():
    '''Parse input options.'''
    import argparse

    parser = argparse.ArgumentParser(description='Convert DOLFIN XML to HDF5.')
    parser.add_argument('infile', type=str, help='input .xml mesh file')

    return parser.parse_args()


def _main():
    ''' Parse command line options and call convert function '''
    import argparse
    parser = argparse.ArgumentParser(description='Convert DOLFIN XML to HDF5.')
    parser.add_argument('infile', type=str, help='input .xml mesh file')
    args = parser.parse_args()

    convert(args.infile)


if __name__ == '__main__':
    _main()
