# CelebA WGAN-GP Project

This project demonstrates the implementation of a Generative Adversarial Network (GAN) using the CelebA face dataset. The GAN is built with PyTorch and utilizes the Wasserstein GAN with Gradient Penalty (WGAN-GP) technique. The project showcase my ability to build a GAN model from different research papers and provides a starting point for further experimentation and improvement.

## Dataset

The dataset used is a subset of the CelebA dataset, which contains facial images. For this project, the dataset was reduced to 5000 images. You can access the dataset [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

## Model Details

- **Base CNN Model**: The CNN architecture used is based on the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1511.06434).
- **Wasserstein GAN**: The GAN framework is based on the [Wasserstein GAN](https://arxiv.org/abs/1701.07875) paper.
- **Wasserstein GAN with Gradient Penalty**: The WGAN-GP technique is implemented following the [Wasserstein GAN with Gradient Penalty](https://arxiv.org/abs/2109.00528) paper.

## Training

- **Epochs**: The model was trained for 200 epochs.
- **Results**: The current results may not be optimal due to the limited number of training epochs and the complexity of the model. The trained generator and discriminator are included for further experimentation.

## Usage

To run the project:

1. Open the provided Jupyter notebook.
2. Load the pre-trained models:
   - `generator.pth`
   - `discriminator.pth`
3. You can also further train the models to improve results. Note that a deeper neural network with more parameters might yield better performance.

## Files Included

- `generator.pth`: The pre-trained generator model.
- `discriminator.pth`: The pre-trained discriminator model.
- `notebook.ipynb`: Jupyter notebook for running and experimenting with the models.

## Notes

- The models may benefit from additional training epochs and/or a more complex neural network architecture to achieve better results.


## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1511.06434)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Wasserstein GAN with Gradient Penalty](https://arxiv.org/abs/2109.00528)
- [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

### Feel free to contribute to this Project.

