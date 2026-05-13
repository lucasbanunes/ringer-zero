import typer
from neuralnet.datasets import app as dataset_app
from neuralnet.models.vkan import app as vkan_app
from neuralnet.models.vqat import app as vqat_app
from neuralnet.models.mlp import app as mlp_app


app = typer.Typer(help="Ringer Zero CLI", rich_markup_mode="markdown")
app.add_typer(vqat_app, name="vqat")
app.add_typer(vkan_app, name="vkan")
app.add_typer(mlp_app, name="mlp")
app.add_typer(dataset_app, name="datasets")

if __name__ == "__main__":
    app()
