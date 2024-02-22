Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, [Latent Diffusion Models (LDMs)](https://arxiv.org/pdf/2112.10752.pdf) quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to LDMs becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

We'll be focusing on the text-to-image case, but I urge you to check out the [original paper](https://arxiv.org/pdf/2112.10752.pdf) which described multiple other use cases, including image inpainting, conditioning on semantic maps, and layout-to-image, with image generation conditioned on annotated bounding boxes.

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (text, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler is ultimately responsible for guiding the vehicle (U-Net) through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt is the closest latent space representation to the final image. The destination latent is then decoded and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through the time-controlled, sequential addition of noise before being reconstructed. The repetitive dismantling teaches the network how to construct coherent and cohesive images, reversing the diffusion process. Importantly, LDMs were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by decomposing training images can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for entities who could accomodate the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space. For that, we need an encoder.

### Autoencoder

Autoencoders have become ubiquitous across machine learning architecture. Their purpose is to receive an input and encode that input to a compressed representation with minimal information loss. The original LDM paper allowed for a pretrained autoencoder to accomplish this task. In practice, most implementations actually trained new autoencoders from scratch to adhere to the paper's loss functions. If you haven't already, I suggest you check out my page [explaining the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet_ML.md). It gives a broad overview of the typical encoder-decoder architecture. Autoencoders for LDMs offer more complexity than the U-Net, but maintain the same principles. 

Encoding to a latent space requires decisions on the size of the latent space. The authors experimented with multiple downsampling factors, ultimately determining that downsampling factors of 4 or 8 offered the best performance. Downsampling factors of 1 or 2 were considered prohibitively expensive, operating near pixel-space, and greatly slowing the training rate. Downsampling by a factor of 16 or more was determined to cause excessive information loss and low fidelity. We'll be focusing on downsampling by a factor of 4 which has more referrals in the original research paper than the factor of 8 model. We'll refer to this specific model as LDM-4 for the remainder of the page.

#### Encoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_encoder_architecture.png" alt="Illustration of LDM encoder" width="100%"
</p>

Constituting the encoder are a variety of convolution operations, ResNet blocks, attention operators, activation functions, and downsampling. [One Stable Diffusion implementation](https://github.com/CompVis/stable-diffusion/tree/main) is illustrated above. Through every step of the encoder operation, the aim is to efficiently condense image features. This is accomplished through an initial convolution, broadening the channels to 128, where pixel-space values are converted to feature embeddings. Immediately after this convolution are a pair of ResNet and attention blocks, providing propagation and self-attention of all image features. This continues to a loop of ResNet, attention blocks, and downsampling. In this loop, a pair of ResNet and attention blocks are applied twice to ensure feature propagation before downsampling. Downsampling reduces the height and width of the data by half, until we have compacted our data to the expected latent dimension size. Since each loop halves the height and width, we repeat the loop log<sub>2</sub>f times. For LDM-4, downsampling would be performed twice, LDM-8 would perform this loop three times. After exiting the loop, a sequence of ResNet-attention-ResNet blocks inspects the condensed image features before normalization stabilizes the data across all channels. A nonlinear activation function is applied elementwise through the Sigmoid function before another convolution operation controls our number of output channels. 

Mention encoder's responsibility in finding perceptually-equivalent lower-dimensional space. First stage of autoencoder is perceptual compression, then comes semantic compression and learning in latent space.

#### KL-regularization

To stabilize the latent space and prevent any pockets of high variance, we apply a low-penalty KL-regularization scheme. [KL-regularization has been shown to be very effective](https://proceedings.neurips.cc/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf) in efficiently unifying diverging distributions. This serves to push the latent space to an approximation of a Gaussian distribution, smoothing out an otherwise unpredictable and high-variance encoding space. Unlike other encoder-decoder architectures, like the U-Net, the latent space for LDMs serves as more than an avenue to image reconstruction. The latent space of LDMs is a legitimate destination in its own right. All diffusion, generation, and denoising at inference time take place in this latent space; stabilizing this space affords a steadier venue for these operations. KL-regularization accomplishes this target, pushing the latent space distribution to an approximate Gaussian, and balancing our latent space.

The original paper also included VQ-GAN regularization, but ultimately KL was determined to provide better results. Include literature supporting this fact.

#### Decoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_decoder_architecture.png" alt="Illustration of LDM decoder" width="100%"
</p>

Our decoder is assembled from the same blocks as our encoder and performs similar functions. Every step of the decoder serves to reconstruct our latent encoding to its pixel-space reconstruction. In the training stage, these operations are focused on refining their weights to offer the most accurate mapping of a lower-dimensional encoding to the higher-dimensional image. Similar to the encoder, our first step is through a convolution block. Broadening the number of channels offers more environments to preserve information while upsampling. The broadened encoding is then passed through a ResNet-attention-ResNet sandwich, self-attending every encoded feature. The filtered representation is then passed through the same loop as our encoder, upsampling rather than downsampling. We upsample log<sub>2</sub>f times to expand our features to their original dimensions. For LDM-4, the features pass through the loop twice. Reaching pixel-space at the end of our loop, we normalize our image before applying the sigmoid function elementwise. Lastly, we pass it through a 3x3 convolution and arrive at our reconstructed image. 

Similar to the encoder, we perform fewer operations at the pixel-space dimensions. The intermediate stages have performed the necessary filtering of image features. Additional operations at higher dimensions could confuse our upsampling parameters and lead to a lower-fidelity reconstruction.

Also, initial ResNet in every loop is controlling number of channels. Also, touch again on importance of ResNet and attention blocks.

#### Metric

Two metrics are used to determine the autoencoder's success: perceptual loss and patch-based adversarial loss. 

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/vgg16_architecture.png" alt="Illustration of a VGG16 convolutional neural network" width="55%"
</p>
  
Perceptual loss measures the semantic understanding of the reconstructed image in comparison to the original. Both the original and reconstructed images are passed through a pre-trained [VGG16](https://www.mygreatlearning.com/blog/introduction-to-vgg16/) convolutional neural network. There are 5 max pooling locations in the VGG16 network (highlighted in red in the image above). The original and reconstructed images' encodings are compared at each of these locations via mean-squared error (MSE). The total MSE is summed across the five output locations to determine the perceptual loss of the reconstructed image. MSE is preferred to typical Euclidean-space losses, such as L2, which depend on pixel-wise comparison. Minimizing the Euclidean distance between two images assumes pixel-wise independence and averages all plausible outputs of the reconstructed image, [encouraging blurring](https://arxiv.org/pdf/1801.03924.pdf). The success of the perceptual loss metric can be [dependent on the network](https://arxiv.org/pdf/2302.04032.pdf) employed for semantic comparisons, and [later literature](https://arxiv.org/pdf/2307.01952.pdf) would demonstrate a decreased emphasis on perceptual loss.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/patch_based_adversarial_loss.jpeg" alt="Example of patch-based adversarial loss" width="40%"
</p>

[Patch-based adversarial loss](https://arxiv.org/pdf/1611.07004.pdf) borrows from GAN theory, introducing controlled patches of noise to reconstructed images while training a discriminator to detect the noisy patches. Introducing localized patches [enforces pixel-space realism](https://arxiv.org/pdf/2012.09841.pdf). Aiding a discriminator in the detection of artificial images by introducing scalar patches of noise encourages the decoder to maintain perceptually important high-frequency details while decoding from the latent space. The patch-based loss is not utilized for an initial chunk of training (50k or so steps), allowing the autoencoder to establish robustness in its encoding-decoding paradigm. Immediately training with both the perceptual and adversarial losses would lead to an overly powerful discriminator and a weaker autoencoder.

### Scheduler

Schedulers are algorithmic guides to the denoising process implemented through the U-Net architecture. Training revolves around learning the additive noise process to understand the guided reversal of noise in an image. For more information on schedulers, I recommend reading the page I wrote focusing on [their literature and implementation evolution](https://github.com/ejohansson13/concepts_explained/blob/main/Stable%20Diffusion/Schedulers_ML.md).

SD does not learn schedules while training, uses plethora of pre-researched scheduling algorithms for sampling. When training model holistically -> intializes previously trained autoencoder. Train autoencoder -> freeze it -> utilize its latent space to train U-Net and conditional model?. Pass latent, conditioning information, and timesteps to compute loss.

Sample noise from normal distribution
Q-sample (additive noise process; add noise to) start latent
  Add t timesteps of noise to start latent
Pass noisy latent, t (telling U-Net how many timesteps of noise were added), and conditioning information to U-Net
U-Net predicts NOISE that was added to latent -> MSE loss between predicted added noise and ground-truth added noise
OR predicts original latent 

I don't actually think you're learning the schedule when training holistically. You're just implementing the learned schedules from previous literature. The U-Net is the only thing actually learning throug training. The U-Net is learning how to implement the scheduling algorithms in predicting noise from images.

Scheduling theory holds up when drawing from perfectly normalized distributions. As much as we regularize and penalize the latent space between our encoder and decoder, it will not be a perfect Gaussian distribution. We use the U-Net to learn how to remove added noise from latents to the best of its ability. When sampling at inference time, we will not be drawing from a perfect distribution, and we will not know the amount of noise added. The U-Net learns, during the holistic training, how best to apply its learned denoising algorithms in the removal of noise from sampled latents. 
Additionally, during training the U-Net is learning how different conditioning prompts result in certain latent space destinations. We're given a picture of a red truck, we add noise to it, we predict how much noise to remove, we're penalized by MSE for how much our predicted denoised latent resembles the original latent, but we're also learning that all latents of red trucks have this constitution. All latents of blue trucks have a similar constitution. Purple monkeys have a completely different latent space destination. U-Net is concurrently learning navigation of latent space denoising through scheduler, while learning latent space destinations through conditioning.

### U-Net

The U-Net is an encoder-decoder architecture popularized through its performance in image segmentation with limited datasets before being recognized as a successful architecture for most computer vision tasks. In LDMs, the U-Net is responsible for the repetitive denoising of the latent. 

Ostensibly doesn't have to be U-Net architecture, quickly became ubiquitous for image generation models. U-Net does seem to downsample latents or at least in DDPM it did.

### Conditioning

Just about any kind of encoder for text/audio/image. Pretrained domain-specific encoder. LDM paper uses BERT encoder. Mention that future research demonstrated that more powerful text encoders often lead to better image generation results.

## Inference

Classifier-free diffusion guidance greatly improves sample quality (not really sure where to put this, probably in training section, right?)

### Scheduler

### U-Net

### Conditioning (Prompt)

Empty conditioning - classifier free guidance, passed through CLIP, used to condition U-Net


### Decoder

# Future Steps

Already seen that increasing size of U-Net leads to improved results (SDXL). Can also add refiner/superresolution U-Net to upsample to more detailed image dimensions. LM performance can have significant effect on downstream image generation performance (Imagen). What other future directions can image synthesis take / what can be done to further improve performance?





# Citations

VGG16 architecture image - https://pub.towardsai.net/the-architecture-and-implementation-of-vgg-16-b050e5a5920b

Patch-based adversarial loss training image - Rao, S., Stutz, D., Schiele, B. (2020). Adversarial Training Against Location-Optimized Adversarial Patches. In: Bartoli, A., Fusiello, A. (eds) Computer Vision – ECCV 2020 Workshops. ECCV 2020. Lecture Notes in Computer Science(), vol 12539. Springer, Cham. https://doi.org/10.1007/978-3-030-68238-5_32
