Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, [Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf) quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to latent diffusion models becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

We'll be focusing on the text-to-image case, but I urge you to check out the [original paper](https://arxiv.org/pdf/2112.10752.pdf) which described multiple other use cases, including image inpainting, conditioning on semantic maps, and layout-to-image, with image generation conditioned on annotated bounding boxes.

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (text, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler is ultimately responsible for guiding the vehicle (U-Net) through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt is the closest latent space representation to the final image. The destination latent is then decoded by the decoder section of the architecture and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through the diffusion process. By sequentially adding noise to images to the point of their destruction, a model can learn to rebuild images from white noise. Importantly, latent diffusion models were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by disassembling training images to noise can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for entities who could accomodate the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space. For that, we need an encoder.

### Autoencoder 15:00 in Aleksa' LDM paper review

Autoencoders have become ubiquitous across machine learning architecture. Their purpose is to receive an input and encode that input to a compressed representation with minimal information loss. The original paper allowed for a pretrained autoencoder to accomplish this task. In practice, most implementations actually trained new autoencoders from scratch to adhere to the paper's loss functions. If you haven't already, I suggest you check out my page [explaining the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet_ML.md). It gives a broad overview of the typical encoder-decoder architecture. Latent diffusion models offer more complexity than the U-Net, but maintain the same principles. 

Encoding to a latent space requires decisions on the size of the latent space. The authors experimented with multiple downsampling factors, ultimately determining that downsampling factors of 4 or 8 offered the best performance. Downsampling factors of 1 or 2 were considered prohibitively expensive, operating near pixel-space, and greatly slowing the training rate. Downsampling by a factor of 16 or more was determined to cause excessive information loss and low fidelity. We'll be focusing on downsampling by a factor of 4 which has more referrals in the original research paper than the factor of 8 model. We'll refer to this specific model as LDM-4 for the remainder of the page.

#### Encoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_encoder_architecture.png" alt="Illustration of LDM encoder" width="100%"
</p>

Constituting the encoder are a variety of convolution operations, ResNet blocks, attention operators, activation functions, and downsampling. [One Stable Diffusion implementation](https://github.com/CompVis/stable-diffusion/tree/main) is illustrated above. Through every step of the encoder operation, the aim is to effectively and efficiently condense image features. This is accomplished through an initial convolution, broadening the channels to 128, where pixel-space values are converted to feature embeddings. Immediately after this convolution are a pair of ResNet and attention blocks, providing propagation and self-attention of all image features. This continues to a loop of ResNet, attention blocks, and downsampling. In this loop, a pair of ResNet and attention blocks are applied twice to ensure feature propagation. Downsampling reduces the height and width of the data by half, until we have compacted our data to the expected latent dimension size. Since each downsampling operation halves the height and width, we would repeat the loop log<sub>2</sub>f times. For LDM-4, downsampling would be performed twice, LDM-8 would perform this loop three times. After exiting this loop, a sequence of ResNet-attention-ResNet blocks ensures analysis of the condensed image features before normalization stabilizes the data across all channels. A nonlinear activation function is applied elementwise through the Sigmoid function before another convolution operation controls our number of output channels. 

Something about the role of encoder to ENCODE image data

#### KL-regularization

To stabilize the latent space and prevent any pockets of high variance, we apply a low-penalty KL-regularization scheme. This serves to push the latent space to an approximation of a Gaussian distribution, smoothing out what could otherwise be an unpredictable and high-variance encoding space. Unlike other encoder-decoder architectures, like the U-Net, the autoencoder is not the final product. It serves as an avenue to a latent space for more efficient calculations. All diffusion, generation, and denoising at inference time take place in this latent space; ensuring an approximate Gaussian distribution for sampling greatly simplifies that process.

[KL-regularization has been shown to effectively](https://proceedings.neurips.cc/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf)...

#### Decoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_decoder_architecture.png" alt="Illustration of LDM decoder" width="100%"
</p>



Decoder: Stacks of convolution, activation functions, Resnet Blocks, upsampling, exactly what you'd find in U-Net.

Conv -> ResNet blocks -> self-attention -> Normalization -> upsampling

#### Metric

Perceptual loss: Compare pre-trained VGG16 network latent representations (multiple (5)) of your encoded-decoded images to the original via MSE. Comparing semantic information captured as opposed to perceptual information which can contain uncooperative high-frequency details.

Patch-based adversarial loss: Intuitive patch-based approach, scalars on patches of matrices tell you whether that patch is real or not. Latent-based patches? Read Pix2Pix to double check.

For first x (50k or so) steps, only perceptual loss is trained. This allows autoencoder to build up some robustness in its encoding prior to adversarial training. No guarantee of successful output if it was immediately trained with adversarial loss (also would be almost reducing it to a GAN). Mention the intuitive explanation for perceptual loss, but also why later literature demonstrated a diminished appetite for it. 

Initial training of adversarial loss: you're telling the discriminator one image is real, one image is fake. Then, you're passing in a generated image and seeing if the discriminator can STILL discern whether it's real or fake. This allows the discriminator to get better at discerning fake images, and the autoencoder to ensure that image reconstruction has high fidelity, enough to fool the discriminator. Two-player game.

Trained by combination of perceptual loss and patch-based adversarial objective. Avoids blurriness introduced by relying solely on pixel-space based losses. All training images in RGB space (HxWx3). Experiment with two kinds of regularizations: Previously popularized VQ and KL. SD models would ultimately use KL-regularization.

### Diffusion (Scheduler)

For more information, I recommend reading the page I wrote focusing on [schedulers](https://github.com/ejohansson13/concepts_explained/blob/main/Stable%20Diffusion/Schedulers_ML.md). The objective of schedulers is to best approximate the denoising score function of an image. In training, they ...

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
