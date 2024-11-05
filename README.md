<div align="center">

# Variational inference for pile-up removal at hadron colliders with diffusion models

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
</div>

This project is generated from the RODEM template for training deep learning models using PyTorch, Lightning, Hydra, and WandB. It is loosely based on the PyTorch Lightning Hydra template by ashleve.

## Submodules

This project relies on a custom submodule called `tools` stored [here](https://gitlab.cern.ch/malgren/tools) on CERN GitLab.
This is a collection of useful functions, layers and networks for deep learning developed by the RODEM group at UNIGE.

If you didn't clone the project with the `--recursive` flag you can pull the submodule using:

```
git submodule update --init --recursive
```

## Docker and Gitlab

This project is setup to use the CERN GitLab CI/CD to automatically build a Docker image based on the `docker/Dockerfile` and `requirements.txt`.
It will also run the pre-commit as part of the pipeline.
To edit this behaviour change `.gitlab-ci`

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or suggestions.
Please use the provided `pre-commit` before making merge requests!

## License

This project is licensed under the MIT License. See the [LICENSE](https://gitlab.cern.ch/rodem/projects/projecttemplate/blob/main/LICENSE) file for details.