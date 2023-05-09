import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# internal
from UNet import UNet
import pipeline as pl 
from tools import misc

class DiffusionModel(nn.Module):
    def __init__(self, diffusion_config, unet_config,
                 device="cuda"):
        super().__init__()
        self.device=device
        self.diffusion_config=diffusion_config
        self.unet_config=unet_config
        self.normalizer = transforms.Normalize(mean=0.4734, std=0.2516)
        
        self.network = UNet(device=self.device, **unet_config)
        self.ema_network = copy.deepcopy(self.network)
        self.optimizer = T.optim.AdamW(self.network.parameters(),
                                       lr=diffusion_config.learning_rate)
        self.loss = nn.MSELoss()
        self.noise_loss_tracker = []
        self.image_loss_tracker = []

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.std**0.5
        return T.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = T.acos(T.tensor(self.diffusion_config.max_signal_rate,
                                      device=self.device))
        end_angle = T.acos(T.tensor(self.diffusion_config.min_signal_rate,
                                    device=self.device))

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = T.cos(diffusion_angles)
        noise_rates = T.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network(noisy_images,noise_rates**2,
        # pred_noises = network([noisy_images, noise_rates**2],
                            #   training=training
                              )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = T.ones((num_images, 1, 1, 1), device=self.device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = T.randn(tuple([num_images]+self.unet_config.input_shape),
                                device=self.device)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        # self.generated_images.permute(0, 3, 1, 2)
        return generated_images.permute(0, 2, 3, 1)
        return generated_images.reshape(num_images, self.unet_config["input_shape"][1],
                                        self.unet_config["input_shape"][2],
                                        self.unet_config["input_shape"][0])

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images)
        noises = T.rand_like(images)

        # sample uniform random diffusion times
        diffusion_times = T.rand(
            size=(len(images), 1, 1, 1),
            device=self.device
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True
        )

        noise_loss = self.loss(noises, pred_noises)  # used for training
        image_loss = self.loss(images, pred_images)  # only used as metric

        # apply gradients
        noise_loss.backward()
        self.optimizer.step()
        noise_loss= noise_loss.cpu().detach().numpy()
        image_loss= image_loss.cpu().detach().numpy()
        self.noise_loss_tracker.append(noise_loss)
        self.image_loss_tracker.append(image_loss)

        # track the exponential moving averages of weights
        # print()
        # print(list(self.network.down_blocks.parameters())[0].sum())
        # print(list(self.ema_network.down_blocks.parameters())[0].sum())
        self.ema_network.exponential_moving_averages(self.network.state_dict(),
                                                     self.diffusion_config.ema)
        # print(list(self.network.down_blocks.parameters())[0].sum())
        # print(list(self.ema_network.down_blocks.parameters())[0].sum())

        return noise_loss #{m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, plot_diffusion_steps=20, epoch=None,
                    logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )
        generated_images = generated_images.cpu().detach().numpy()
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index], vmax=1, vmin=0)
                plt.axis("off")
        plt.tight_layout()
        plt.title(epoch)
        plt.show()
        plt.close()
        return generated_images

if __name__ == "__main__":
    config = misc.load_yaml("configs/configs.yaml")
    # load dataset
    # train_dataset = pl.prepare_dataset(config.dataset_name,
    #                                 config.dataset_repetitions,
    #                                 config.batch_size,
    #                                 "train[:80%]+validation[:80%]+test[:80%]")
    # val_dataset = pl.prepare_dataset(config.dataset_name,
    #                                 config.dataset_repetitions,
    #                                 config.batch_size,
    #                                 "train[80%:]+validation[80%:]+test[80%:]")
    train_sample = pl.ImagePipeline(config.dataset_name)

    # run diffusion
    model = DiffusionModel(diffusion_config=config.diffusion_config,
                           unet_config=config.unet_config,
                           device=config.device)
    dataloader = train_sample.get_dataloader()
    loss=[]
    for ep in range(config.num_epochs):
        for i in tqdm(dataloader):
            images = T.tensor(i.numpy(), device=config.device)
            loss.append(model.train_step(images))
            # break
        generated_images = model.plot_images(epoch=ep)
        if np.any(np.isnan(generated_images)):
            break
    
    plt.figure()
    plt.plot(loss)