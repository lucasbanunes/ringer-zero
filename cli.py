import typer
from ringer_zero.models.vqat_q7 import app as vqatq7_app


app = typer.Typer()
app.add_typer(vqatq7_app, name='vqat_q7')

if __name__ == '__main__':
    app()
