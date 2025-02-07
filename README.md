## flux sliders

修改自<https://github.com/rohitgandikota/sliders>

代码仍在快速迭代中，目标是兼容diffuers库，可以与现有flux生态整合。

### 安装环境

```bash
conda env create -f environment.yml
```

### 训练

直接执行以下命令，会训练person放大和缩小的slider lora。

```bash
python train_flux_concept_sliders.py
```

### 推理

```bash
python predict.py
```

### 推理结果

![demo](assets/demo.jpg)
