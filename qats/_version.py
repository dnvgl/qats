from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qats")
except PackageNotFoundError:
    __version__ = ""
