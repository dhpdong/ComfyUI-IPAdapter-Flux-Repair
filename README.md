# ComfyUI-IPAdapter-Flux-Repair
Repair the issue for OOM.
Primitive nodes may have some models that cannot be cleaned up during multiple inferences, occupying memory and ultimately leading to OOM. We have fixed this issue.

You need to use the modified model loader and sampler nodes. Please refer to the following diagram for the connection method.
![fcf48be6-d7d6-4515-8e6d-260de9fab95a](https://github.com/user-attachments/assets/4bca6600-527f-4756-87a5-aefe7a11602b)

Attention: After the repair, when repeatedly inferring the same image, it will start from the model loading node, which will result in longer inference time. Currently, there is no good solution.
