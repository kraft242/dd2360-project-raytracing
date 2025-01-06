DD2360 Project
==================================

This is the repository for our project in the course DD2360 HT24 at KTH Royal Institute of Technology.
We are: 
- Lennard Scheibel (lsche@kth.se)
- Karl Törnblom Bartholf (karltb@kth.se) 
- Emil Olausson (emilola@kth.se)

The project is forked from a repository by rogerallen ([Original](https://github.com/rogerallen/raytracinginoneweekendincuda)).

Data Types
----------

We tested the following data types:
- fp32 (default)
- fp64
- fp16
- bf16

Each branch is available under a branch in this repository under the corresponding name, i.e `$ git checkout float_colab` to go to the fp32 branch.

Compiling and Running
---------------------

The project includes a `Makefile`. 

Make sure to change the compute version in the variable `GENCODE_FLAGS` before compiling.

Then run `make out.jpg` to create the output image. Make sure you have `ppmtojpeg` installed before running.
