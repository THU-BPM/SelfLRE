
## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

## Usage
* Run the full model on SemEval dataset with default hyperparameter settings<br>

```python3 src/train.py```<br>


## Data
### Format
Each dataset is a folder under the ```./data``` folder:
```
./data
└── SemEval
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json

```
### Download

* SemEval: SemEval 2010 Task 8 data (included in ```data/SemEval```)<br>
* TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))<br>

Then use the scripts from ```data/data_prepare.py``` to further preprocess the data. For SemEval, the script split the original training data into two sets. For TACRED, the script first perform some preprocessing to ensure the same format as SemEval.

