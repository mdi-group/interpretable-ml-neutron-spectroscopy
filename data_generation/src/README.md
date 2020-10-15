# Source files to generate training datasets

These are the (Matlab) source files for the code to generate the training datasets.

To compile these files you need access to Matlab and the Matlab Compiler toolbox.

You also have to initialise the 4 submodules `Horace`, `Herbert`, `spinw` and `brillem`.
If you did not clone this repository with the `--recurse-submodules` option and those directories are empty, you need to do:

```
git submodule init Horace Herbert spinw brillem
git submodule update --remote
```

Within Matlab you can run the `compile_python.m` script to compile the Matlab code which generates the precomputed resolution datasets (compile to a Python module).

In a console, you can run the `compile_it.sh` script to compile the Matlab code which generates the Monte Carlo resolution function datasets (compile to a stand-alone program).
