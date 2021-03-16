
## Task A
### Preprocess data

Generate [gulped dataset](https://github.com/TwentyBN/GulpIO) from the extracted frames
```bash
python taskA/pre_process.py <path_to_frames> <pickle_file> <output_dir>
```
See [here](https://github.com/epic-kitchens/epic-kitchens-55-annotations) for example annotations
The output_dir contains the gulped data that is consumed by the eval.py script


## Model performance evaluation
The command below compares model performance between the [TSN](https://github.com/yjxiong/temporal-segment-networks) and [TRN](https://github.com/zhoubolei/TRN-pytorch). The losses, accuracy and FPS comparision between the two models are displayed in bar plot for visualization

```bash
python eval.py <path_to_gulped_data>
```


