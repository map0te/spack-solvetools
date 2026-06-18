# spack-solvetools

A Spack extension providing solver tools.

## Installation

```console
$ unset SPACK_PYTHON  # ensure Spack uses the Python from the virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install "git+https://github.com/map0te/spack-solvetools#egg=spack_solvetools"
```

## Usage

### List intermediate models

To list all intermediate models during spec solving:

```console
$ spack solvetools list-models zlib
$ spack solvetools list-models -o models.txt "python@3.9 +shared"
$ spack solvetools list-models --fresh hdf5
```

### Profile solver performance

To profile the solve phase and print statistics:

```console
$ spack solvetools profile --timers --stats zlib
$ spack solvetools profile --show=solutions "python@3.9"
$ spack solvetools profile hdf5
```

### Capture solve results

To capture optimization criteria and DAG output for multiple specs:

```console
$ spack solvetools solve-compare run specs.txt -o output-dir --label baseline -j 4
```

To show results from a previous run:

```console
$ spack solvetools solve-compare show output-dir/baseline
$ spack solvetools solve-compare show output-dir/baseline --spec hdf5
```

The `solve-compare` command:
- Runs solves in parallel using Python's `concurrent.futures` (works on Mac and Linux)
- Captures optimization criteria with priorities and values
- Saves DAG output with color codes and highlights non-default variants/versions
- Stores results in JSON format for easy analysis
