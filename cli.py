import typer
from ringer_zero.datasets import app as dataset_app


app = typer.Typer()
app.add_typer(dataset_app, name='datasets')

if __name__ == '__main__':
    app()
