import typer
from ringer_zero.datasets import app as dataset_app
from ringer_zero.models.vqat import app as vqat_app
from ringer_zero.models.mlp import app as mlp_app


app = typer.Typer(
    help='Ringer Zero CLI',
    rich_markup_mode="markdown"
)
app.add_typer(dataset_app, name='datasets')
app.add_typer(mlp_app, name='mlp')
app.add_typer(vqat_app, name='vqat')


if __name__ == '__main__':
    app()
