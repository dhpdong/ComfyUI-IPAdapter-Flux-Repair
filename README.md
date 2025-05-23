# ComfyUI-IPAdapter-Flux-Repair
修复IPAdapter-Flux显存溢出的问题。

IPAdapter-Flux节点，在进行多次推理或两个节点轮换推理时，可能有部分显存无法被正常清理，最终导致显存溢出。本项目对这个问题进行了修复。

你需要同时下载IPAdapter-Flux和本项目，在工作流中用新的GGUF模型导入节点和采样器节点，模型导入节点需要连接一个随机种子生成节点，同时新的采样器节点需要从IPAdapter推理节点的model输出直接连接，其他的连接方式没有改变。具体可以参考下图：
![15e2e9c6-8674-438c-be1e-5ffb39720799](https://github.com/user-attachments/assets/cd661ee3-d089-4d8c-9f3d-ccf5d3ce4da7)

注意：使用修复节点后，在对单张图片重复推理时，第二次开始每次都会从GGUF模型导入节点开始而非采样节点本身，这会导致一些推理时间上的增加，但目前尚无解决方案，还请理解。
