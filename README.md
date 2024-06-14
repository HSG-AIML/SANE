[TODO](): Add overview figure
# SANE: Scalable and Versatile Weight Space Learning

This repository contains the code for the paper "Towards Scalable and Versatile Weight Space Learning," presented at ICML 2024. This work introduces SANE, a novel approach for learning task-agnostic representations of neural networks that are scalable to larger models and applicable to various tasks.

## Summary

### Abstract
Learning representations of well-trained neural network models holds the promise to provide an understanding of the inner workings of those models. However, previous work has faced limitations when processing larger networks or was task-specific to either discriminative or generative tasks. This paper introduces the SANE approach to weight-space learning. SANE overcomes previous limitations by learning task-agnostic representations of neural networks that are scalable to larger models of varying architectures and show capabilities beyond a single task. Our method extends the idea of hyper-representations towards sequential processing of subsets of neural network weights, allowing one to embed larger neural networks as a set of tokens into the learned representation space. SANE reveals global model information from layer-wise embeddings and can sequentially generate unseen neural network models, which was unattainable with previous hyper-representation learning methods. Extensive empirical evaluation demonstrates that SANE matches or exceeds state-of-the-art performance on several weight representation learning benchmarks, particularly in initialization for new tasks and larger ResNet architectures.

### Key Methods
- **Sequential Decomposition**: Breaking down neural network weights into smaller, manageable subsets (tokens) for embedding.
- **Self-Supervised Pretraining**: Using a self-supervised approach to pretrain the SANE model on a variety of tasks and architectures.
- **Model Sampling**: Generating new neural network models by sampling from the learned representation space.

### Results
- **Model Property Prediction**: SANE embeddings demonstrate high predictive performance for model properties such as test accuracy, epoch, and generalization gap across various datasets and architectures.
- **Generative Capabilities**: SANE can generate high-performing neural network models from scratch or fine-tune them with significantly less computational effort compared to training from scratch.
- **Scalability**: The method scales to large models like ResNet-18, preserving meaningful information across long sequences of tokens.

## Code Structure

- **data/**: Scripts for data preprocessing and loading.
- **models/**: Implementations of neural network architectures and the SANE model.
- **training/**: Training scripts for pretraining SANE and fine-tuning on downstream tasks.
- **sampling/**: Scripts for generating new models using the SANE embeddings.
- **utils/**: Utility functions for various tasks such as evaluation, logging, and checkpointing.

## Running Experiments

### Pretraining SANE
To pretrain SANE on a dataset:
```bash
python training/pretrain_sane.py --config configs/pretrain_config.yaml
```



### Fine-Tuning on Downstream Tasks
To fine-tune a pretrained SANE model on a specific task:
```bash
python training/fine_tune.py --config configs/finetune_config.yaml
```


### Generating Models
To generate new models using the pretrained SANE embeddings:
```bash
python sampling/generate_models.py --config configs/generate_config.yaml
```



### Evaluating Model Performance
To evaluate the performance of generated models:
```bash
python utils/evaluate.py --config configs/evaluate_config.yaml
```



## Figures
- **Figure 1**: Overview of SANE architecture and tokenization process.
![Figure 1 Placeholder](path_to_figure_1.png)

- **Figure 2**: Comparison of SANE with other methods on model property prediction.
![Figure 2 Placeholder](path_to_figure_2.png)

- **Figure 3**: Generative capabilities of SANE.
![Figure 3 Placeholder](path_to_figure_3.png)

## Citation
If you use this code in your research, please cite our paper:
```
@inproceedings{schuerholt2024sane,
    title={Towards Scalable and Versatile Weight Space Learning},
    author={Konstantin Sch{"u}rholt and Michael W. Mahoney and Damian Borth},
    booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
    year={2024},
    organization={PMLR}
}
```

## License
This project is licensed under the MIT License.
