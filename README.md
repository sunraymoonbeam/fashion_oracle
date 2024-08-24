# Outfit Classifier

## Usage

1. Install dependencies via pip: `pip install -r requirements.txt`

### Training

1. Please put the dataset inside `./data/raw` (or change `conf/process_data.yaml`)
2. `python src/process_data.py`
3. `python src/train_model.py`

### Inference

1. TBD

## Dataset

TBD

## Modeling

* MobileNetV2


export PYTHONPATH=$(pwd)/src:$PYTHONPATH