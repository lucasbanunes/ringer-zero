from typing import Any, Callable, Iterable, Generator
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import uproot
import awkward as ak
import pandas as pd
import pyarrow as pa
import ROOT

from .utils import walk_paths

ROOT_TO_ARROW_SCALARS = {
    "bool": pa.bool_,
    "Bool_t": pa.bool_,
    "char": pa.int8,
    "Char_t": pa.int8,
    "signed char": pa.int8,
    "unsigned char": pa.uint8,
    "UChar_t": pa.uint8,
    "short": pa.int16,
    "Short_t": pa.int16,
    "unsigned short": pa.uint16,
    "UShort_t": pa.uint16,
    "int": pa.int32,
    "Int_t": pa.int32,
    "unsigned int": pa.uint32,
    "UInt_t": pa.uint32,
    "long": pa.int64,
    "Long_t": pa.int64,
    "unsigned long": pa.uint64,
    "ULong_t": pa.uint64,
    "long long": pa.int64,
    "Long64_t": pa.int64,
    "unsigned long long": pa.uint64,
    "ULong64_t": pa.uint64,
    "float": pa.float32,
    "Float_t": pa.float32,
    "double": pa.float64,
    "Double_t": pa.float64,
    "string": pa.string,
    "std::string": pa.string,
}


def read_ttree_as_ak(
    input_file: str | Path | Iterable[Path] | Iterable[str],
    ttree_name: str = "CollectionTree",
) -> ak.Array:
    """
    Reads a single ttree as an awkward array.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str]
        The path to the input root file(s).
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    ak.Array
        An awkward array containing the data.
    """
    uproot_path = f"{str(input_file)}:{ttree_name}"
    with uproot.open(uproot_path) as ttree:
        ak_array = ttree.arrays(library="ak")
    return ak_array


def read_ttree_as_pdf(
    input_file: str | Path | Iterable[Path] | Iterable[str],
    ttree_name: str = "CollectionTree",
) -> pd.DataFrame:
    """
    Reads a single ttree as a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str]
        The path to the input root file(s).
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data with proper PyArrow dtypes.
    """
    ak_array = read_ttree_as_ak(input_file, ttree_name)
    # It is easier and more consistent to do this than to use ak.to_dataframe
    with TemporaryDirectory() as tmp_dir:
        parquet_path = f"{tmp_dir}/temp.parquet"
        ak.to_parquet(ak_array, parquet_path, list_to32=True)
        df = pd.read_parquet(parquet_path, dtype_backend="pyarrow", engine="pyarrow")
    return df


def read_ttree_as_arrow(
    input_file: str | Path | Iterable[Path] | Iterable[str],
    ttree_name: str = "CollectionTree",
) -> pa.Table:
    """
    Reads a single ttree as a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str]
        The path to the input root file(s).
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data with proper PyArrow dtypes.
    """
    ak_array = read_ttree_as_ak(input_file, ttree_name)
    # It is easier and more consistent to write parquet and read with pyarrow than to use ak.to_arrow_table directly
    with TemporaryDirectory() as tmp_dir:
        parquet_path = f"{tmp_dir}/temp.parquet"
        ak.to_parquet(ak_array, parquet_path, list_to32=True)
        arrow_table = pa.parquet.read_table(parquet_path)
    return arrow_table


def get_rdf_schema(rdf: ROOT.RDataFrame) -> dict[str, str]:
    """
    Extracts the schema of a ROOT RDataFrame as a dictionary.

    Parameters
    ----------
    rdf : ROOT.RDataFrame
        The RDataFrame from which to extract the schema.

    Returns
    -------
    dict
        A dictionary mapping column names to their corresponding ROOT types.
    """
    schema = {}
    for column in rdf.GetColumnNames():
        column_type = rdf.GetColumnType(column)
        schema[str(column)] = column_type
    return schema


def extract_angle_brackets_content(text: str) -> str | None:
    """
    Extract content inside the first balanced <...>, supporting nesting.

    Examples:
    extract_angle_content("ROOT::VecOps::RVec<int>") -> "int"
    extract_angle_content("ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>>")
        -> "ROOT::VecOps::RVec<int>"

    Parameters
    ----------
    text : str
        The input string from which to extract the content. It should contain at least one '<' character.

    Returns
    -------
    str | None
        The content inside the first balanced angle brackets, or None if no valid content is found.
    """
    start = text.find("<")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
            if depth < 0:
                return None  # malformed (more '>' than '<')

    return None  # malformed (unclosed '<')


VECTOR_PREFIXES = ("std::vector<", "vector<", "ROOT::VecOps::RVec<")


ROOT_TO_PYTHON_SCALARS = {
    "bool": pa.bool_,
    "Bool_t": pa.bool_,
    "char": pa.int8,
    "Char_t": pa.int8,
    "signed char": pa.int8,
    "unsigned char": pa.uint8,
    "UChar_t": pa.uint8,
    "short": pa.int16,
    "Short_t": pa.int16,
    "unsigned short": pa.uint16,
    "UShort_t": pa.uint16,
    "int": pa.int32,
    "Int_t": pa.int32,
    "unsigned int": pa.uint32,
    "UInt_t": pa.uint32,
    "long": pa.int64,
    "Long_t": pa.int64,
    "unsigned long": pa.uint64,
    "ULong_t": pa.uint64,
    "long long": pa.int64,
    "Long64_t": pa.int64,
    "unsigned long long": pa.uint64,
    "ULong64_t": pa.uint64,
    "float": pa.float32,
    "Float_t": pa.float32,
    "double": pa.float64,
    "Double_t": pa.float64,
    "string": pa.string,
    "std::string": pa.string,
}


def root_type_to_pyarrow_type(type_name: str, strict: bool = True) -> pa.DataType:
    """
    Convert a ROOT type string to the equivalent PyArrow data type.

    Parameters
    ----------
    type_name : str
        ROOT type name as returned by RDataFrame.GetColumnType.
    strict : bool, optional
        If True, raise ValueError for unsupported types. If False,
        unsupported types are mapped to pa.string().

    Returns
    -------
    pa.DataType
        The mapped PyArrow data type.
    """
    normalized = type_name.strip()
    normalized = normalized.replace("const ", "").replace("&", "").strip()
    if normalized.startswith(VECTOR_PREFIXES):
        inner = extract_angle_brackets_content(normalized)
        return pa.list_(root_type_to_pyarrow_type(inner, strict=strict))

    fixed_array_match = re.match(r"^(.+)\[(\d+)\]$", normalized)
    if fixed_array_match:
        inner_name = fixed_array_match.group(1).strip()
        array_size = int(fixed_array_match.group(2))
        return pa.list_(
            root_type_to_pyarrow_type(inner_name, strict=strict),
            list_size=array_size,
        )

    if normalized in ROOT_TO_ARROW_SCALARS:
        return ROOT_TO_ARROW_SCALARS[normalized]()

    if strict:
        raise ValueError(f"Unsupported ROOT type for PyArrow conversion: {type_name}")
    return pa.string()


def rdf_schema_to_pyarrow_schema(
    rdf_schema: dict[str, str],
    *,
    strict: bool = True,
    nullable: bool = True,
    preserve_root_type_metadata: bool = True,
    metadata_encoding: str = "utf-8",
) -> pa.Schema:
    """
    Convert a ROOT RDataFrame schema dictionary into a PyArrow schema.

    Parameters
    ----------
    rdf_schema : dict[str, str]
        Mapping from column name to ROOT type string.
    strict : bool, optional
        If True, fail on unsupported ROOT types. If False, unsupported
        types are stored as pa.string().
    nullable : bool, optional
        Nullability applied to all generated fields.
    preserve_root_type_metadata : bool, optional
        If True, each field stores the original ROOT type in metadata under
        the key 'root_type'.
    metadata_encoding : str, optional
        Encoding used for metadata values.

    Returns
    -------
    pa.Schema
        A valid PyArrow schema representing the input ROOT schema.
    """
    fields: list[pa.Field] = []
    for column_name, root_type in rdf_schema.items():
        arrow_type = root_type_to_pyarrow_type(root_type, strict=strict)
        metadata = None
        if preserve_root_type_metadata:
            metadata = {"root_type": root_type.encode(metadata_encoding)}

        fields.append(
            pa.field(
                column_name,
                arrow_type,
                nullable=nullable,
                metadata=metadata,
            )
        )

    return pa.schema(fields)


NOCAST_ROOT_TYPES = [
    "Float_t",
    "Double_t",
    "Int_t",
    "int",
    "float",
    "double",
]


def get_root_to_python_caster(root_type: str) -> Callable[[Any], Any]:
    if root_type in NOCAST_ROOT_TYPES:
        return lambda x: x  # No casting, use original type
    elif root_type.startswith(VECTOR_PREFIXES):
        inner_root_type = extract_angle_brackets_content(root_type)
        if inner_root_type is None:
            raise ValueError(f"Malformed vector type: {root_type}")
        inner_caster = get_root_to_python_caster(inner_root_type)
        # Recursively cast elements in the vector
        return lambda x: [inner_caster(elem) for elem in x]
    else:
        raise ValueError(f"Unsupported ROOT type for Python conversion: {root_type}")


def ttree_python_sample_generator(
    input_file: str | Path | Iterable[Path] | Iterable[str],
    ttree_name: str = "CollectionTree",
) -> Generator[dict[str, Any], None, None]:
    for filepath in walk_paths(input_file, file_ext="root"):
        rdf = ROOT.RDataFrame(ttree_name, str(filepath))
        rdf_schema = get_rdf_schema(rdf)
        tree_caster = {
            column_name: get_root_to_python_caster(root_type)
            for column_name, root_type in rdf_schema.items()
        }
        with ROOT.TFile(str(filepath)) as root_file:
            tree = root_file[ttree_name]
            for sample in tree:
                casted_samples = {
                    column_name: caster(getattr(sample, column_name))
                    for column_name, caster in tree_caster.items()
                }
                yield casted_samples


def read_ttree_as_dict(
    input_file: str | Path | Iterable[Path] | Iterable[str],
    ttree_name: str = "CollectionTree",
) -> pd.DataFrame:
    """
    Reads a single ttree as a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str]
        The path to the input root file(s).
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data with proper PyArrow dtypes.
    """
    uproot_path = f"{str(input_file)}:{ttree_name}"
    with uproot.open(uproot_path) as ttree:
        arrow_table = ak.to_arrow_table(ttree.arrays(library="ak"))
        pdf = arrow_table.to_pandas()
    return pdf
