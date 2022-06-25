# tf-cyclegan-music-style-transfer

Refactoring CycleGAN for Music Style Transfer

Original repo: https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization


# Usage


## Preparing the data
To prep the data, place a folder in this directory and create subdirectories labelled according to the genre of of music present in each subdirectory. Example:
```
datasets
    |
    |-- pop
    |   |-- track1.midi
    |   |-- track2.midi
    |   |-- ...
    |
    |-- jazz
        |-- track1.midi
        |-- track2.midi
        |-- ...
```
Then run `src/prepare_data.py`. The basic usage will be to pass your root directory (datasets in the example above) along with the genre subdirectory. For example

```
python prepare_data.py datasets pop
```

The resulting train and test files will be saved under `datasets/tfrecord`. 

### Note:
The script `npy_to_tfrecord.py` is an auxiliary script to convert already existing `.npy` files to tfrecords. It is not a necessary step but could be of use.

## Training

Change the model parameter options as needed in `src/config.yaml` and you're good to go! 

The training script only requires two arguments: `path_a` and `path_b` which are the paths to the tfrecord of genres A and B.
```
python src/train.py <path_to_a> <path_to_b>
```

Optional parameters (and default values) are:
```
    --batch_size (32)
    --epochs (5)
    --model_output ("trained_models")
    --config_path ("src/config.yaml")
    --log_dir ("train_logs")
    --learning_rate (0.0002)
    --lr_step (3)
    --beta1 (0.5)
```

The resulting model will be saved in the location indicated by `--model_output`. Training logs can be viewed with TensorBoard `tensorboard --logdir train_logs`.


## Converting new samples

To convert songs from one genre you first need to have run the `prepare data` step. 
```
python src/convert.py --model_path <path_to_model> --outpath <path_to_output>
```
The output folder will contain three outputs for each sample and are indexed in the order they appear in the dataset:
```
    {idx}_original.mid -> The original sample
    {idx}_transfer.mid -> The converted sample
    {idx}_cycle.mid    -> Result of converting the transferred sample back to the original style
```
