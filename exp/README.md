
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
python taskA/eval.py <path_to_gulped_data>
```
## TaskB
### Preprocess data
Generate gulped dataset for training and validation.
sub_folders should be named the class for images, and the corresponding images should be placed in the sub_folders<br/>
To generate gulped dataset for training, 

```bash
python taskB/pre_process.py <path_to_training_images> <output_dir>
```
### Peform training..
```bash
python taskB/train.py -h

positional arguments:
  train_gulp            path to training gulped data
  val_gulp              path to val gulped data

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --lr LR, --learning_rate LR
                        learning rate
  --b B, --batch_size B



```
In order to view accuracy and loss curves between train and val 
```bash
tensoboard --logdir results/
```
During training, the weights that produce the lowest validation loss is stored ```checkpoint.pt```
The train script also shows a plot-window after training is complete with the loss and acc plots which can be saved for review.

### Inference

```bash
python taskB/inference.py -h

positional arguments:
  weights     weights file for model
  video_file  path to video file
  json_file   json file containing index to class mappings

optional arguments:
  -h, --help  show this help message and exit

```
saves video with predicted labels on frame in ```output.avi```




