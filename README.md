# keras_dqn_snake

# Introduction
Deep Q Network keras implemention to play Gluttonous Snake.

# Requirements
python3.6
- `numpy`
- `keras (tensorflow backend)`
- `opencv-python`

# Usage
Use --help to see usage of main.py.
```
usage: main.py [-h] [--mode MODE] [--path PATH]

optional arguments:
  -h, --help   show this help message and exit
  --mode MODE  train or test model, default "train"
  --path PATH  path to model weight file, default ./model/dqn_model.h5
  ```

# Display
`cd keras_dqn_snake`

To train dqn model, run:

`python main.py --mode "train"`

To test dqn model, run:

`python main.py --mode "test"`
