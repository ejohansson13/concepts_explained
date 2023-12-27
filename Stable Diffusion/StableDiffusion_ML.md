Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Something about previous architectures, success of transformer for nlp and other ml tasks. Parallel to sucess of diffusion models for image generation. Now, shift to multimodal models.

# Architecure

Overarching encoder-decoder architecure. In between, you've got three main stages: diffusion model, denoising u-net, and conditioner (prompt: written, image, audio, multimodal capabilities). Encoder is responsible for encoding images to their more compact latent representations. Diffusion model is repsonsible for learning how to take images apart, adding Gaussian noise at sequential timesteps to understand how noise alters images and eventually how the process can be reversed. Scheduler is responsible for picking out a latent representation from the Gaussian noise cesspit that is created from our diffusion model. U-Net is responsible for taking noisy, latent representation from scheduler and denoising it (still maintaining latent space, maximizing efficiency) such that our image can quickly be generated with some further conditioning. The conditioning comes via our prompt. Considering the text-to-image synthesis example, we feed in an empty prompt for classifier-free guidance of our model before feeding in our wanted textual prompt. The empty and textual prompts are concatenated together and combined with our U-Net result via the cross-attention mechanism popularized from the Transformer architecture. This gives us our latent end-product. This latent space matrix is then passed through our decoder, upsampled, etc. etc. etc. to become a legitimate pixel-space image. Following that, you can pass it through super-resolution networks or whatever else you want to do.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through the diffusion process. By sequentially adding noise to images to the point of their destruction, a model can learn to rebuild images from white noise. Importantly, latent diffusion models were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by disassembling a variety of training images to noise can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures (cite some here) operated on the pixel-space of images, reserving their utility for the largest companies who could afford the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space. For that, we need an encoder.

### Encoder

### Diffusion

### U-Net

### Conditioning

### Decoder

### Metric

## Inference

### Scheduler

### U-Net

### Prompt

Empty conditioning - important

### Decoder
