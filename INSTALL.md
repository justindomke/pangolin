# Just want to test out pangolin?

Assuming you have [uv](https://docs.astral.sh/uv/) installed, you can download pangolin and start python with pangolin in a temporary environment with this one-liner:

```shell
uv run --with pangolin python
```

# Want to install it "properly"?

Install pangolin and all dependencies using your preferred tool:

* **pip**: `pip install pangolin`
* **uv (project)**: `uv add pangolin`
* **uv (pip interface)**: `uv pip install pangolin`
* **Poetry**: `poetry add pangolin`
* **pipenv**: `pipenv install pangolin`

Note that PyTorch is not installed by default, because PyTorch is huge, can be tricky to install. If you want to use the PyTorch backend you can either just install pytorch yourself (recommended) or you can install it along with pangolin by using one of these commands:

* **pip**: `pip install pangolin[pytorch]`
* **uv (project)**: `uv add pangolin[pytorch]`
* **uv (pip interface)**: `uv pip install pangolin[pytorch]`
* **Poetry**: `poetry add pangolin[pytorch]`
* **pipenv**: `pipenv install pangolin[pytorch]`
