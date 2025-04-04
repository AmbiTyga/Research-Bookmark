# ReFT
Training ReFT for VLM

## Setup
We will be using `Ovis2-4b`. It has this padding side bug, we have fixed that, but since it comes with the weights, we are pushing the code in here. To run the training, download the weights using following command:
```
huggingface-cli download AIDC-AI/Ovis2-4B --local-dir $PWD/Ovis2-4b --local-dir-use-symlinks False --include "*.safetensors"
```