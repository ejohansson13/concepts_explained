Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, latent diffusion models quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to [Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf) becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

Something about how in this page, we'll focus on the text-to-image case. But urge readers to check out original research paper because it describes so many other cool achievements, like layout-to-image.

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (text, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler is ultimately responsible for guiding the vehicle (U-Net) through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt is the closest latent space representation to the final image. The destination latent is then decoded by the decoder section of the architecture and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through the diffusion process. By sequentially adding noise to images to the point of their destruction, a model can learn to rebuild images from white noise. Importantly, latent diffusion models were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by disassembling training images to noise can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for the largest companies who could afford the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space. For that, we need an encoder.

### Autoencoder 15:00 in Aleksa' LDM paper review


Pre-trained autoencoders. 

Encoder: Stacks of convolution (3x3 kernels!!), activation function, downsampling blocks, exactly what you'd find in a U-Net. Downsampling is by predefined factor f! Double check what Resnet blocks are. In latent space of encoder, image features are actually self-attended to. Latent representations are "unrolled" as feature embeddings, then self-attention is performed. "Every token attends to every other token".

Conv -> ResNet blocks -> self-attention -> Normalization -> downsampling

#### Decoder

Decoder: Stacks of convolution, activation functions, Resnet Blocks, upsampling, exactly what you'd find in U-Net.

Conv -> ResNet blocks -> self-attention -> Normalization -> upsampling

Perceptual loss: Compare pre-trained VGG16 network latent representations (multiple (5)) of your encoded-decoded images to the original via MSE. Comparing semantic information captured as opposed to perceptual information which can contain uncooperative high-frequency details.

Patch-based adversarial loss: Intuitive patch-based approach, scalars on patches of matrices tell you whether that patch is real or not. Latent-based patches? Read Pix2Pix to double check.

Downsampling factor f = 1,2 results in slow training progress. Leaves most of perceptual compression responsibility to diffusion model. f > 16 causes stagnating fidelity quickly after training. Exaggerated initial compression results in information loss. Significant FID gap between pixel-based diffusion and LDM-8 after 2M training steps.

#### Metric

For first x (50k or so) steps, only perceptual loss is trained. This allows autoencoder to build up some robustness in its encoding prior to adversarial training. No guarantee of successful output if it was immediately trained with adversarial loss (also would be almost reducing it to a GAN).

Initial training of adversarial loss: you're telling the discriminator one image is real, one image is fake. Then, you're passing in a generated image and seeing if the discriminator can STILL discern whether it's real or fake. This allows the discriminator to get better at discerning fake images, and the autoencoder to ensure that image reconstruction has high fidelity, enough to fool the discriminator. Two-player game.

Trained by combination of perceptual loss and patch-based adversarial objective. Avoids blurriness introduced by relying solely on pixel-space based losses. All training images in RGB space (HxWx3). Experiment with two kinds of regularizations: Previously popularized VQ and KL. SD models would ultimately use KL-regularization.

KL-regularization is applied to latent space of autoencoder. "Smooths" out these latent representations. Ensures that we have faithful latent encodings for our latent space. Remember, all diffusion and denoising takes place in latent space. For that reason, we want to make sure our autoencoder has capable latent encodings.

Have to read https://proceedings.neurips.cc/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf, 
We are getting posteriors - encoded latents
Get moments from posterior (mean, variance, std. dev?)
"Create" Gaussian distribution from those moments (posterior)
Compute KL divergence of these posteriors
Regularize towards "normal" Gaussian distribution according to KL-divergence.
Keep in mind this is only a slight KL-divergence loss.

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
