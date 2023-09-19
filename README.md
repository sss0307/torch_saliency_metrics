# torch_saliency_metrics

## Evaluation Metrics for Visual Saliency Prediction Models

Welcome to the Evaluation Metrics section for Visual Saliency Prediction Models. This repository provides Python files that enable the evaluation of visual saliency prediction models using metrics inspired by the popular MIT Saliency Benchmark.

### Introduction

Visual Saliency Prediction models are used to predict regions of interest or salient regions in images or videos, simulating human visual attention. Evaluating the performance of these models is crucial for understanding their effectiveness in various applications, such as image and video analysis.

This repository offers a versatile evaluation toolset, including a class called `Evaluation_metrics()`, designed to assess the quality of saliency predictions against ground truth fixation maps.

### Getting Started

Before using the evaluation metrics provided here, make sure you have the following:

1. **PyTorch Dataloaders**: You should have two PyTorch Dataloaders prepared:
   - One containing human fixation maps (ground truth).
   - Another containing model-generated saliency maps for evaluation.

### Usage

To use the evaluation metrics in your project, follow these steps:

1. Import the `EvaluationMetrics` class into your Python script after copying the file into your script folder.

```python
from evals_ import Evaluation_metrics
```

2. Create instances of the `EvaluationMetrics` class, specifying the names of your dataloaders for human fixation maps and model-generated saliency maps.

```python
eval_metrics = Evaluation_metrics(human_dataloader, model_dataloader)
```

### Contributing

We welcome contributions to improve and extend the functionality of this evaluation metrics toolset. If you have ideas for enhancements or bug fixes, please feel free to open an issue or submit a pull request.

### License

This repository is provided under the [MIT License](LICENSE), allowing you to use and modify the code for your evaluation needs.

### Acknowledgments

This toolset is inspired by the MIT Saliency Benchmark and builds on the work of researchers in the field of visual saliency prediction. We acknowledge their contributions to the advancement of this area of computer vision.

Thank you for using our evaluation metrics for visual saliency prediction models. We hope it proves valuable in your research and applications.
