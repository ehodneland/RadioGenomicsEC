import os


def get(fullpath):

    dirname, filename = os.path.split(fullpath)
    basename, ext = os.path.splitext(filename)
    return dirname, filename, basename, ext
