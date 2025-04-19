import os
import subprocess
from pathlib import Path

import pytest

for root, _dirs, files in os.walk("examples"):
    examples = [Path(root, file) for file in files if file.endswith(".py")]


@pytest.mark.parametrize("file", examples)
def test_run_example(file: Path) -> None:
    subprocess.run([file], check=True)
