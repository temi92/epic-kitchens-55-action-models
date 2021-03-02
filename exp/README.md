
## Preprocess data
Generate [gulped dataset] (https://github.com/TwentyBN/GulpIO) from the extracted frames
```bash
python pre_process.py <path_to_frames> <path_to_pickle_label> <output_dir>
```
The output_dir contains the gulped data that is consumed by the eval.py script


## Model performance evaluation
The command below compares model performance between the [TSN] (https://github.com/yjxiong/temporal-segment-networks) and [TRN](https://github.com/zhoubolei/TRN-pytorch). The losses, accuracy and FPS comparision between the two models are displayed in bar plot for visualization

```bash
python eval.py <path_to_gulped_data>
```