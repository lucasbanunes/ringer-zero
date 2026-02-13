import typer
from ringer_zero.models.vqat0 import app as vqat0_app


app = typer.Typer()
app.add_typer(vqat0_app, name='vqat0')

if __name__ == '__main__':
    app()
