# pybbn
Bayesian Belief Networks using Python.

# Installation Prerequisites

1. `cairo`: `brew install cairo`
2. `tkinter`: `brew install python-tk`
3. `gsn2x`: Download `macOS` binary from [https://github.com/jonasthewolf/gsn2x/releases](https://github.com/jonasthewolf/gsn2x/releases).

# Open source libraries used

## [`gsn2x`](https://github.com/jonasthewolf/gsn2x)
A program that renders [Goal Structuring Notation](https://scsc.uk/gsn) in a YAML format to a scalable vector graphics (SVG) image.

### Ubuntu 20.04 (gsn2x - v2.8.0)

Refer to [this discussion thread](https://github.com/jonasthewolf/gsn2x/discussions/333) for installing the required binaries in Ubuntu 20.04.
- Clone the repo and run `cargo build --release`.
- The author of the project suggests cloning the latest `2.8` version tag.
- Navigate to `gsn2x-xx-xx-xx/target/releases` and copy the executable to where you want.
- Run syntax: `gsn2x <.yaml file>`

## [`pybbn`](https://py-bbn.readthedocs.io/index.html)

`pybbn` is a Python implementation of probabilistic and causal inference in Bayesian Belief Networks using exact inference algorithms. You may install `pybbn` from [pypi](https://pypi.org/project/pybbn/).

```shell
pip install pybbn
```

## `customtkinter`
