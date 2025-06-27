# CellCon-GI
Code for the paper **'Cellular Morphology-Conditioned Generative Inpainting for Atypical Mitosis Classification'**

Abstract:
```
Atypical mitoses are critical prognostic markers for tumor proliferation, yet classification
efforts suffer from class imbalance, data scarcity, and noisy labels. We address these challenges
with a novel approach for biologically semantic inpainting, conditioned on a histological context
patch, an inpainting mask, and a chromosome segmentation mask. This triple-conditioned generative
strategy allows to disentangle the mitosis shape information from the cellular context and enables
the utilization of large-scale datasets that do not contain atypical subclassification and hence
allows to train generative approaches on much larger dataset scales. We evaluate both adversarial
and diffusion-based inpainting strategies. Our approach effectively mitigates the lack of data
diversity in training and label noise, thereby substantially improving classification performance
for atypical vs. normal mitoses (0.865 vs. 0.789 in balanced accuracy). Additionally, we release
two new multi-domain atypical vs. normal mitosis datasets, used as hold out sets in our study,
to support research in this underexplored field.
```

## Pipeline overview
![Atypical_MICCAI_Figure1](https://github.com/user-attachments/assets/6f72dda6-9892-4209-a97e-7ff04f641421)

## Usage notes

Please find the code in the folders: 
- [inpainting_models] Both inpainting models
- [downstream_classification] EfficientNet-B0-based classification task
- [LoRA] LoRA fine-tuned foundation model based downstream classification


