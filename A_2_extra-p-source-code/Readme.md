Extra-P
=======

Extra-P is a performance modeling tool which we extended to create differential performance models.

Installation
------------

Extra-P requires an installation of Python 3.8 including pip to install it. The first step is to copy the extrap folder.
Next we recommend using a virtual envirionment to run Extra-P. The installation of Extra-P can simply be performed by calling
`pip install <path to the Extra-P folder>`.

### Setup an virtual environment to test/develop the package

1. `python -m venv venv` Create a new virtual python environment to test/develop the code.
2. Activate the virtual environment to use it for testing/developing.
    * On Windows, use `venv\Scripts\activate.bat`
    * On Unix or macOS, use `source venv/bin/activate`
3. `deactivate` Deactivate the virtual environment.

### Install the Extra-P package from a local src tree

`pip install -e <path>` installs package in developer mode from a local src tree via sym links. If you are already in
the root folder, you can use `pip install -e .`

Run Extra-P
-----------

Run `extrap` to start the command line version of Extra-P. You can find several example datasets in
the [tests/data](extrap/tests/data) folder. 
Run `extrap-gui` to start the graphical user interface.

Run Extra-P tests
-----------------

The tests in the [tests](extrap/tests) folder can be run with Python's unittest module or with PyTest. We recommend using
PyTest to execute the tests in Extra-P.

### PyTest

1. `cd tests` change your working directory to the tests folder
2. Run `pytest` to start the test execution
    * By default, the tests need an installed GUI environment. If you want to omit the GUI tests (e.g., for CI) you can
      use the following option: `--ignore-glob=test_gui*.py`

### Python unittest module

The following steps are necessary to use the unittest module.

1. Add the root folder to the `PYTHONPATH`
2. Change your working directory to the `tests` folder
3. Run the unittest module to start the test execution `python -m unittest`

Create Differential Performance Models
--------------------------------------

The differential performance modeling process works in the Extra-P GUI works as follows. 
First of all, you load and if necessary create the models for your first set of measurements. 
Than you choose the *Compare experiments* action in the *File* menu. 
This opens the guided step-by-step process for experiment comparison. 
Once the process is finished, Extra-P will display the unified call-tree of both experiments.
Using the *Plots* menu you can open the *Difference plot* and the *Comparison plot*.