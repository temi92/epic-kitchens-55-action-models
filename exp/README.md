
## Task A
### Preprocess data

Generate [gulped dataset](https://github.com/TwentyBN/GulpIO) from the extracted frames
```bash
python taskA/pre_process.py <path_to_frames> <pickle_file> <output_dir>
```
See [here](https://github.com/epic-kitchens/epic-kitchens-55-annotations) for example annotations<br/>
The output_dir contains the gulped data that is consumed by the eval.py script


### Model performance evaluation
The command below compares model performance between the [TSN](https://github.com/yjxiong/temporal-segment-networks) and [TRN](https://github.com/zhoubolei/TRN-pytorch). The losses, accuracy and FPS comparision between the two models are displayed in bar plot for visualization

```bash
python eval.py <path_to_gulped_data>
```
## TaskB
### Preprocess data
Generate gulped dataset for training and validation
For example to generate gulped dataset for training, 

```bash
python taskB/pre_process.py <path_to_training> <output_dir>
```
### Peform training..
```bash
python taskB/train.py <train_gulp> <val_gulp>

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --lr LR, --learning_rate LR
                        learning rate
  --b B, --batch_size B

```




