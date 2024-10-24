import os

import pytest

examples = []
for root, _dirs, files in os.walk("examples"):
    for file in files:
        if file.endswith(".py"):
            examples.append(os.path.join(root, file))


@pytest.mark.parametrize("file", examples)
def test_run_example(file):
    assert os.system(file) == 0
