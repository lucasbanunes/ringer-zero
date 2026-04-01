import typer
from ringer_zero.datasets import app as dataset_app
from ringer_zero.models.vkan import app as vkan_app
from ringer_zero.models.vqat import app as vqat_app


app = typer.Typer(help="Ringer Zero CLI", rich_markup_mode="markdown")
app.add_typer(vqat_app, name="vqat")
app.add_typer(vkan_app, name="vkan")
app.add_typer(dataset_app, name="datasets")

if __name__ == "__main__":
    app()
