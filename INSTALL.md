# How to install pangolin

These instructions will install pangolin and all dependencies in a local directory, using the [uv](https://docs.astral.sh/uv/) package manager.

**Step 1** If you haven't installed [uv](https://docs.astral.sh/uv/), install it.

**Step 2** Download pangolin source code and extract it. You could do this in either of the following ways:

1. Type the following at the command line (assumes you have `wget` and `unzip` installed):

    ```
    cd path/to/wherever/
    wget https://github.com/justindomke/pangolin/archive/refs/heads/main.zip
    unzip main.zip
    ```

2.  Go to [`https://github.com/justindomke/pangolin`](https://github.com/justindomke/pangolin) in your browser, then click on "Code" and then "Download zip". Then double-click the file to extract it and past the contents wherever you want to store them.

**Step 3.** Go to the directory and install pangolin and all dependencies.

```
uv sync
```

Note that PyTorch is not installed by default. If you want to use the PyTorch backend, you can either use `uv sync —extra torch` to try to install torch automatically. Or, if that doesn't work for you for whatever reason, you can just do `uv sync` and then install pytorch locally with `uv pip install torch functorch`.

**Step 4.** Start Python make sure Pangolin works.

```
% python
Python 3.12.4
Type "help", "copyright", "credits" or "license" for more information.
>>> from pangolin import interface as pi
>>> x = pi.normal(0,1)
>>> pi.print_upstream(x)
```

You should see something like this:

```
shape | statement
----- | ---------
()    | a = 0
()    | b = 1
()    | c ~ normal(a,b)
```