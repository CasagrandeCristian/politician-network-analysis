
# Network Analysis

Coursework exploring network structure, community detection, information
propagation, and network robustness using Python and NetworkX. The repository
contains the code, input data, experiment results, visualizations, and reports
for three assignments.

## Assignments

- **Assignment 1 - Network metrics and communities:** structural measures,
  scale-free analysis, Louvain/greedy/label-propagation community detection,
  and comparisons between partitions.
- **Assignment 2 - Information propagation:** simulations of correct and
  altered information under different thresholds and configurations.
- **Assignment 3 - Network robustness:** giant-component removal experiments
  on synthetic and politician-page networks, including targeted attacks and
  reinforcement strategies.

Each assignment folder contains its Python scripts and copies of the input
network data under `code/`, experiment outputs under `csv/` (where applicable),
figures under `png/`, and the written report under `pdf/`.

## Dataset

The Facebook Pages Politician network is from
[Network Repository](https://networkrepository.com/fb-pages-politician.php).
It contains approximately 5.9k nodes and 41.7k edges. The node and edge CSV
files are included in each assignment's `code/` directory so the scripts can
be run independently.

Please refer to the dataset source for its terms of use and attribution
requirements.

## Requirements

Python 3 and the following packages are used:

- `matplotlib`
- `networkx`
- `numpy`
- `pandas`
- `powerlaw`
- `scikit-learn`

Install them with:

```bash
python -m pip install matplotlib networkx numpy pandas powerlaw scikit-learn
```

## Running the scripts

Run scripts from their assignment's `code/` directory because they load the
CSV files using paths relative to the current working directory. For example:

```bash
cd Assignment1/code
python part1Metrics.py
python part2Part3Part4Community.py
```

The scripts in Assignments 2 and 3 follow the same pattern. Depending on the
script, they print results to the terminal and/or write CSV files and plots to
the current working directory. The committed `csv/` and `png/` directories
contain example outputs from the completed experiments.

## Acknowledgements

ChatGPT was used as a support tool to explore analysis methods, develop parts
of the plotting and data-processing logic, and review the English writing.
Overleaf was used to create and format the LaTeX reports.

---
