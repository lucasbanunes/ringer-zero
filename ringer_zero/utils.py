from typing import Iterable, Iterator
from pathlib import Path


def walk_paths(
        paths: str | Path | Iterable[str | Path],
        file_ext: str,
        dev: bool = False) -> Iterator[Path]:
    """
    Generator that opens all directories in an iterator for
    a specific file extension. This is useful for script cases where
    an user can pass a mix of directories and filepaths.

    Parameters
    ----------
    paths : str | Path | Iterable[str | Path]
        A single path or an iterable of paths. These can be directories or
        file paths. If a directory is provided, it will search recursively
        for files with the specified file extension.
    file_ext : str
        The desired file extension to look for
    dev: bool
        If True, the function will yield just the first file found

    Yields
    ------
    Path
        The path to a file

    Raises
    ------
    ValueError
        Raised if there is a file that does not have file_ext as its extension
    """
    if isinstance(paths, str):
        paths = [Path(paths)]
    elif isinstance(paths, Path):
        paths = [paths]
    i = 0
    for ipath in paths:
        if ipath.is_file():
            if ipath.suffix != f'.{file_ext}':
                raise ValueError(
                    f'File {ipath} does not have the expected extension .{file_ext}')
            yield ipath
            i += 1
            if dev and i > 0:
                break
        else:
            for filepath in ipath.glob(f'**/*.{file_ext}'):
                yield filepath
                i += 1
                if dev and i > 0:
                    break
