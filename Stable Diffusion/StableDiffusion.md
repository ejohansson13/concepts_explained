Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, latent diffusion models quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to [Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf) becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

Something about how in this page, we'll focus on the text-to-image case. But urge readers to check out original research paper because it describes so many other cool achievements, like layout-to-image.

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (text, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler is ultimately responsible for guiding the vehicle (U-Net) through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt is the closest latent space representation to the final image. The destination latent is then decoded by the decoder section of the architecture and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through the diffusion process. By sequentially adding noise to images to the point of their destruction, a model can learn to rebuild images from white noise. Importantly, latent diffusion models were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by disassembling training images to noise can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for the largest companies who could afford the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space. For that, we need an encoder.

### Autoencoder 15:00 in Aleksa' LDM paper review

Pre-trained autoencoders. 

Encoder: Stacks of convolution (3x3 kernels!!), activation function, downsampling blocks, exactly what you'd find in a U-Net. Downsampling is by predefined factor f! Double check what Resnet blocks are.

Decoder: Stacks of convolution, activation functions, Resnet Blocks, upsampling, exactly what you'd find in U-Net.

Downsampling factor f = 1,2 results in slow training progress. Leaves most of perceptual compression responsibility to diffusion model. f > 16 causes stagnating fidelity quickly after training. Exaggerated initial compression results in information loss. Significant FID gap between pixel-based diffusion and LDM-8 after 2M training steps.

#### Metric

Trained by combination of perceptual loss and patch-based adversarial objective. Avoids blurriness introduced by relying solely on pixel-space based losses. All training images in RGB space (HxWx3). Experiment with two kinds of regularizations: Previously popularized VQ and KL. SD models would ultimately use KL-regularization.

### Diffusion (Scheduler)

Refer to schedulers.md

### U-Net

Does in fact operate on the latent space, see reweighted bound in LDM paper. Latent representations can be quickly obtained from Encoder through training, and latent representations can be quickly passed through Decoder in a single pass.

### Conditioning

Just about any kind of encoder for text/audio/image. Pretrained domain-specific encoder. LDM paper uses BERT encoder. Mention that future research demonstrated that more powerful text encoders often lead to better image generation results.

## Inference

Classifier-free diffusion guidance greatly improves sample quality (not really sure where to put this, probably in training section, right?)

### Scheduler

### U-Net

### Conditioning (Prompt)

Empty conditioning - important

### Decoder

# Future Steps

Already seen that increasing size of U-Net leads to improved results (SDXL). Can also add refiner/superresolution U-Net to upsample to more detailed image dimensions. LM performance can have significant effect on downstream image generation performance (Imagen). What other future directions can image synthesis take / what can be done to further improve performance?
