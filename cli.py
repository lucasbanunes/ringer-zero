import typer
from ringer_zero.datasets import app as dataset_app
from ringer_zero.models.vqat import app as vqat_app


app = typer.Typer()
app.add_typer(vqat_app, name='vqat')
app.add_typer(dataset_app, name='datasets')

if __name__ == '__main__':
    app()
