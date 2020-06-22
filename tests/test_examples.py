import pytest

import os

examples = []
for root, dirs, files in os.walk("examples"):
    for file in files:
        if file.endswith(".py"):
            examples.append(os.path.join(root, file))


@pytest.mark.parametrize('file', examples)
def test_run_example(file):
    assert os.system(file) == 0