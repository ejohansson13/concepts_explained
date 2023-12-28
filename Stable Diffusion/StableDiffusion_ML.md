Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, latent diffusion models quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and autoregressive modeling which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to latent diffusion models becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: a diffusion model, the denoising u-net, and the custom conditioner (written, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their more compact latent representation. The diffusion process is repsonsible for taking images apart, adding Gaussian noise at sequential timesteps. **read research papers on schedulers** 
Scheduler is responsible for picking out a latent representation from the Gaussian noise cesspit that is created from our diffusion model. **read research papers on schedulers**
Our U-Net is responsible for taking the noisy, latent representation from the scheduler and denoising it. The U-Net carries the bulk of the responsibility during training.
such that our image can quickly be generated with some further conditioning. The conditioning comes via our prompt. Considering the text-to-image synthesis example, we feed in an empty prompt for classifier-free guidance of our model before feeding in our wanted textual prompt. The empty and textual prompts are concatenated together and combined with our U-Net result via the cross-attention mechanism popularized from the Transformer architecture. This gives us our latent end-product. This latent space matrix is then passed through our decoder, upsampled, etc. etc. etc. to become a legitimate pixel-space image. Following that, you can pass it through super-resolution networks or whatever else you want to do.

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

# Future Steps

Already seen that increasing size of U-Net leads to improved results (SDXL). Can also add refiner/superresolution U-Net to upsample to more detailed image dimensions. LM performance can have significant effect on downstream image generation performance (Imagen). What other future directions can image synthesis take / what can be done to further improve performance?
