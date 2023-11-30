from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("qats")
except PackageNotFoundError:
    __version__ = ""
