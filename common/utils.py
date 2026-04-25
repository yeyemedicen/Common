''' utils module
    for functions of general use and utility
'''
import git
import os
import shutil


def is_Expression(obj):
    ''' Check if object is a dolfinx fem.Function (replaces legacy Expression). '''
    from dolfinx import fem
    return isinstance(obj, fem.Function)


def is_Constant(obj):
    ''' Check if object is a dolfinx fem.Constant. '''
    from dolfinx import fem
    return isinstance(obj, fem.Constant)


def is_enriched(V):
    ''' Check if FunctionSpace V uses an enriched (P1b / bubble) element. '''
    return hasattr(V.ufl_element(), '_elements')


def on_cluster():
    import platform
    import re
    uname = platform.uname()
    pattern = re.compile("^(leftraru\d)|(cn\d\d\d)")
    return bool(pattern.match(uname[1]))


def prep_mesh(mesh_file):
    if on_cluster():
        # running on NLHPC cluster
        # TODO: UNTESTED CHANGE. Test this.
        # mfile = '/home/dnolte/fenics/nitsche/meshes/' + mesh_file
        mfile = shutil.os.getcwd() + '/' + mesh_file
        mesh = '/dev/shm/' + mesh_file
        shutil.copy(mfile, mesh)
    else:
        # running on local machine
        mesh = mesh_file

    return mesh


def trymkdir(path):
    ''' mkdir if path does not exist yet '''
    import os
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_git_rev_hash(path):
    ''' Get the rev hash of the git repository that 'path' (a file or
    directory) is located in.

    Args:
        path    directory or file inside a git repository

    Returns:
        hex_sha of HEAD
    '''
    git_repo = git.Repo(os.path.dirname(os.path.realpath(path)),
                        search_parent_directories=True)
    return git_repo.head.object.name_rev
