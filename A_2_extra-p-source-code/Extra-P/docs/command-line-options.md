Extra-P Command Line Options
============================

The Extra-P command line interface has the following options.

| Arguments                                                                                                                                           |                                                                                                                                                                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Positional**                                                                                                                                      |                                                                                                                                                                                                         |
| _FILEPATH_                                                                                                                                          | Specify a file path for Extra-P to work with                                                                                                                                                            |
| **Optional**                                                                                                                                        |                                                                                                                                                                                                         |
| `-h`, `--help`                                                                                                                                      | Show help message and exit                                                                                                                                                                              |
| `--version`                                                                                                                                         | Show program's version number and exit                                                                                                                                                                  |
| `--log` {`debug`, `info`, `warning`, `error`, `critical`}                                                                                           | Set program's log level (default: `warning`)                                                                                                                                                            |
| **Input options**                                                                                                                                   |                                                                                                                                                                                                         |
| `--cube`                                                                                                                                            | Load a set of CUBE files and generate a new experiment                                                                                                                                                  |
| `--extra-prof`                                                                                                                                      | Load a set of ExtraProf files and generate a new experiment                                                                                                                                             |
| `--extra-p-3`                                                                                                                                       | Load data from Extra-P 3 (legacy) experiment                                                                                                                                                            |
| `--json`                                                                                                                                            | Load data from JSON or JSON Lines input file                                                                                                                                                            |
| `--nsight`                                                                                                                                          | Load a set of Nsight files and generate a new experiment (EXPERIMENTAL)                                                                                                                                 |
| `--talpas`                                                                                                                                          | Load data from Talpas data format                                                                                                                                                                       |
| `--text`                                                                                                                                            | Load data from text input file                                                                                                                                                                          |
| `--experiment`                                                                                                                                      | Load Extra-P experiment and generate new models                                                                                                                                                         |
| `--scaling` {`weak`, `weak_parallel`, `strong`}                                                                                                     | Set scaling type when loading data from per-thread/per-rank files (CUBE files) (default: weak)                                                                                                          |
| `--keep-values`                                                                                                                                     | Keeps the original values after import                                                                                                                                                                  |
| **Modeling options**                                                                                                                                |                                                                                                                                                                                                         |
| `--median`                                                                                                                                          | Use median values for computation instead of mean values                                                                                                                                                |
| `--modeler` {`default`, `basic`, `refining`, `multi-parameter`}                                                                                     | Selects the modeler for generating the performance models                                                                                                                                               |
| `--options` _KEY_=_VALUE_ [_KEY_=_VALUE_ ...]                                                                                                       | Options for the selected modeler                                                                                                                                                                        |
| `--help-modeler` {`default`, `basic`, `refining`, `multi-parameter`}                                                                                | Show help for modeler options and exit                                                                                                                                                                  |
| **Output options**                                                                                                                                  |                                                                                                                                                                                                         |
| `--out` _OUTPUT_PATH_                                                                                                                               | Specify the output path for Extra-P results                                                                                                                                                             |
| `--print` {`all`,`all-python`,`all-latex`, `callpaths`, `metrics`, `parameters`, `functions`,`functions-python`,`functions-latex`, _FORMAT_STRING_} | Set which information should be displayed after modeling. Use one of the presets or specify a formatting string using placeholders (see [output-formatting.md](output-formatting.md)). (default: `all`) |
| `--save-experiment` <i>EXPERIMENT_PATH</i>                                                                                                          | Saves the experiment including all models as Extra-P experiment (if no extension is specified, “.extra-p” is appended)                                                                                  |
| `--model-set-name` _NAME_                                                                                                                           | Set the name of the generated set of models when outputting an experiment (default: “New model”)                                                                                                        |
| `--disable-progress`                                                                                                                                | Disables the progress bar outputs of Extra-P to stdout and stderr                                                                                                                                       |                                                                                                                             
