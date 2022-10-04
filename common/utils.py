''' utils module
    for functions of general use and utility
'''
import git
import os
import shutil
import dolfin
import ufl


def is_enriched(V):
    ''' Check if the given function space or sub function space has enriched
    elements. '''

    # FIXME pybind11 hack for >=2018.1.0:
    if (dolfin.__version__ >= '2018'
            and isinstance(V, dolfin.cpp.function.FunctionSpace)):
        V = dolfin.FunctionSpace(V)

    while V.num_sub_spaces():
        V = V.sub(0)
    return isinstance(V.ufl_element(),
                      ufl.finiteelement.enrichedelement.EnrichedElement)


def is_Expression(obj):
    ''' Check if object has type dolfin Expression '''
    if dolfin.__version__ >= '2018':
        return isinstance(obj, dolfin.function.expression.Expression)
    else:
        return isinstance(obj, dolfin.functions.expression.Expression)


def is_Constant(obj):
    ''' Check if object has type dolfin Constant '''
    if dolfin.__version__ >= '2018':
        return isinstance(obj, dolfin.function.constant.Constant)
    else:
        return isinstance(obj, dolfin.functions.constant.Constant)


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
