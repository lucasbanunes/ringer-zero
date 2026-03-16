from pathlib import Path
from typing import Annotated
import duckdb
import typer


app = typer.Typer()


@app.command()
def print_schema(
    dataset_dir: Annotated[
        Path,
        typer.Option('--dataset-dir', help='Directory containing the dataset files')
    ],
):
    for table in dataset_dir.glob('*.parquet'):
        with duckdb.connect(':memory:') as conn:
            res = conn.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{str(table)}/*.parquet')").fetch_df()
            print(20*'-')
            print(f"Schema for {table.name}:")
            print(res.to_string())
            print(20*'-')
