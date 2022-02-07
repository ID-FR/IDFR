# IDFR

A keras implement of Enhenced countering adversarial attack via input denoising and feature restoring.

- `attack`: the script to generate adversarial examples.
- `train`: the model structure and training process of IDFR.

## Attention

1. All the package version is in `version.txt`.
2. To speed up the training, the adversarial samples were saved and read directly during the training instead of being generated. Since the data is read using absolute paths, please modify the code to your local counterpart, and the creation process of the adversarial samples is in the attack folder. It can be summarized as reading clean samples, generating adversarial samples, and storing data and labels as npy format files.
3. convex hull look here: https://github.com/ID-FR/IDFR/blob/main/train/optim.py#L290-L303

## Others

In train folder, I've put up the code in jupyter lab format, which has the result output.
