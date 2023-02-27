# PSL Runs for Visual Sudoku Puzzle Classification

This repository covers the experiments for the paper [Visual Sudoku Puzzle Classification: A Suite of Collective Neuro-Symbolic Tasks](https://linqs.org/publications/#id:augustine-nesy22).
These scripts were run and tested on Linux.

## Running Experiments

To run experiments, you first have to choose the data you would like to use.
 - All data is available [here](https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/).
 - [This repository](https://github.com/linqs/visual-sudoku-puzzle-classification) contains pointers to specific data chunks.
 - Sample datasets that only have one split: [without overlap](https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/sample_4x4.zip) and [with overlap](https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/sample_4x4_overlap.zip).

Note that most of these scripts have configurable paths (check with `--help`),
but this README will use the defaults.

### Setting up the Data

After you get the data, we now need to put it in the `data` directory.
Move your data (starting with the `dimension::*` dir(s)) into the `data/raw` directory.

For example the sample dataset (with overlap) above will make a data directory like:
```
data
└── raw
    └── dimension::4
        └── datasets::mnist
            └── strategy::simple
                └── numTrain::00100
                    └── numTest::00100
                        └── numValid::00100
                            └── corruptChance::0.50
                                └── overlap::1.00
                                    └── split::01
                                        ├── options.json
                                        ├── test_cell_labels.txt
                                        ├── test_puzzle_labels.txt
                                        ├── test_puzzle_notes.txt
                                        ├── test_puzzle_pixels.txt
                                        ├── train_cell_labels.txt
                                        ├── train_puzzle_labels.txt
                                        ├── train_puzzle_notes.txt
                                        ├── train_puzzle_pixels.txt
                                        ├── valid_cell_labels.txt
                                        ├── valid_puzzle_labels.txt
                                        ├── valid_puzzle_notes.txt
                                        └── valid_puzzle_pixels.txt
```

Now, generate the PSL data from the raw data:
```
./scripts/convert-data.py
```

Your data directory should now include another directory named `vspc` where all the PSL data has been generated:
```
data
└── vspc
    └── dimension::4
        └── datasets::mnist
            └── strategy::simple
                └── numTrain::00100
                    └── numTest::00100
                        └── numValid::00100
                            └── corruptChance::0.50
                                └── overlap::1.00
                                    └── split::01
                                        ├── digit_model_untrained.h5
                                        ├── digit_model_untrained_tf
                                        │   ├── assets
                                        │   ├── fingerprint.pb
                                        │   ├── saved_model.pb
                                        │   └── variables
                                        │       ├── variables.data-00000-of-00001
                                        │       └── variables.index
                                        ├── eval
                                        │   ├── digit_features.txt
                                        │   ├── digit_labels.txt
                                        │   ├── digit_targets.txt
                                        │   ├── digit_truth.txt
                                        │   ├── label_id_map.txt
                                        │   ├── positive_digit_features.txt
                                        │   ├── positive_digit_targets.txt
                                        │   ├── positive_digit_truth.txt
                                        │   ├── positive_row_col_violation_targets.txt
                                        │   ├── positive_violation_targets.txt
                                        │   ├── positive_violation_truth.txt
                                        │   ├── row_col_violation_targets.txt
                                        │   ├── violation_targets.txt
                                        │   └── violation_truth.txt
                                        ├── learn
                                        │   ├── digit_features.txt
                                        │   ├── digit_labels.txt
                                        │   ├── digit_targets.txt
                                        │   ├── digit_truth.txt
                                        │   ├── first_positive_puzzle.txt
                                        │   ├── label_id_map.txt
                                        │   ├── positive_digit_features.txt
                                        │   ├── positive_digit_pinned_truth.txt
                                        │   ├── positive_digit_targets.txt
                                        │   ├── positive_digit_truth.txt
                                        │   ├── positive_row_col_violation_targets.txt
                                        │   ├── positive_violation_targets.txt
                                        │   ├── positive_violation_truth.txt
                                        │   ├── row_col_violation_targets.txt
                                        │   ├── violation_targets.txt
                                        │   └── violation_truth.txt
                                        └── options.json
```

### Running Baselines

Now you can run the baselines with:
```
./scripts/run-baselines.py
```

All results will be stored in the `results` directory.
(The results directory is checked if an experiment has already been run before running it again.)

### Running NeuPSL

To run NeuPSL, use:
```
./scripts/run-psl.sh
```

All results will be stored in the `results` directory.

### Looking at Results

All results are stored (by default) in the `results` directory.

You an use the `parse-results` script to pull out most the values you should need:
```
./scripts/parse-results.py
```
