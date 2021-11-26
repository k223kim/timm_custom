# timm_custom

## Description

In the case of your data having only 1 channel while want to use timm models (with or without pretrained weights), run the following command to get the model with appropriate number of input channels.

`python load_timm.py --model "densenet" --model_depth 121 --pretrained True --num_classes=4 --n_input_channels 1`

## Argument

The argument explanation is below:

```
usage: load_timm.py [-h] [--model MODEL] [--model_depth MODEL_DEPTH]
                    [--pretrained PRETRAINED] [--num_classes NUM_CLASSES]
                    [--n_input_channels N_INPUT_CHANNELS]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name
  --model_depth MODEL_DEPTH
                        Depth of the model
  --pretrained PRETRAINED
                        If true, will use ImageNet pretrained weight
  --num_classes NUM_CLASSES
                        number of classes
  --n_input_channels N_INPUT_CHANNELS
                        number of input channels
```

Notice that the script will check if the given model + model_depth is in timm models. Additionally, if `pretrained=True`, it will check if that particular model has a pretraiend weight or not. If not, it will just set `pretrained=False`.

## Usage

In the case of a project where it handles CT images, it only has 1 channel since there is no RGB channel.

## Example

When running `python load_timm.py --model "densenet" --model_depth 121 --pretrained True --num_classes=4 --n_input_channels 3`, the model architecture is shown below:

```
DenseNet(
  (features): Sequential(
    (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm0): BatchNormAct2d(
      64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
      (act): ReLU(inplace=True)
    )
...
```

When running `python load_timm.py --model "densenet" --model_depth 121 --pretrained True --num_classes=4 --n_input_channels 1`, the model architecture has been updated as shown below:

```
DenseNet(
  (features): Sequential(
    (conv0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm0): BatchNormAct2d(
      64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
      (act): ReLU(inplace=True)
    )
...
```
## Warning

Since timm models pretrained weights are trained on the ImageNet dataset (which has 3 channels), it will lose the "initial benefit" of pretrained weights.
