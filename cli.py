import typer
from ringer_zero.models.vqat0 import app as vqat0_app
from ringer_zero.models.vnoq0 import app as vnoq0_app


app = typer.Typer()
app.add_typer(vqat0_app, name='vqat0')
app.add_typer(vnoq0_app, name='vnoq0')

if __name__ == '__main__':
    app()
