from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field
import submitit


class SlurmConfig(BaseModel):

    logs_dir: Annotated[
        Path,
        Field(
            description='Directory to save the logs of the slurm jobs'
        )
    ]

    def get_executor(self) -> submitit.AutoExecutor:
        return submitit.AutoExecutor(folder=str(self.logs_dir))
