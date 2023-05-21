# ResNet 2bit MNIST

Using PACT implementation to achieve $ \ge99.6 $ % accuracy on MNIST dataset with ResNet model.

## Results
Current achievements - ResNet 2bit with accuracy = 99.69% \
You can check results in `model_performance.ipynb` file

Set constant `PATH_TO_MNIST` in `model_performance.ipynb` \
Or change `download` argument in EMNIST initialization to true

Model architecture - ResNet44 (implementation in `resnet.py`) \
Checkpoint (model weights) - `checkpoints/BestModel/BestModel.pt`