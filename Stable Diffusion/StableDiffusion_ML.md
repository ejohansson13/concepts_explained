Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, [Latent Diffusion Models (LDMs)](https://arxiv.org/pdf/2112.10752.pdf) quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to LDMs becoming the eminent architecture for text-to-image generation tasks. Their compatibility with natural language, image, and audio prompts (thanks to cross-attention) accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

We'll be focusing on the text-to-image case, but I urge you to check out the [original paper](https://arxiv.org/pdf/2112.10752.pdf) which described multiple other use cases, including image inpainting, conditioning on semantic maps, and layout-to-image, with image generation conditioned on annotated bounding boxes.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_diagram.png" alt="LDM diagram taken from original research paper" width="100%"
</p>

# Architecure

The latent diffusion network follows an encoder-decoder architecure. Between these blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (text, image, or audio prompt for multimodal capabilities). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler guides the U-Net through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt is the closest latent space representation to the final image. The destination latent is then decoded and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. We pump in a large dataset of images for our network to learn a variety of visual themes and configurations. These images are taken apart through a forward process consisting of the time-controlled, sequential addition of Gaussian noise to a training image. This repetitive destruction of images teaches the network the probability distribution of destroying images but also the conditional probability of generating coherent images from random noise. The process of generating lucid images from noise is referred to as the reverse diffusion process. Importantly, LDMs were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by decomposing training images can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for entities who could accomodate the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_graph.png" alt="Graph demonstrating perceptual vs semantic compression training tradeoff taken from 2022 paper on latent diffusion models" width="50%">
</p>

The above graph demonstrates that the majority of bits constituting a digital image correspond to high-frequency, perceptual details. In contrast, relatively few bits comprise the semantic information of the image. Unlike previous diffusion models that tried to balance (cite here) multiple loss terms prioritizing the perceptual and semantic training, latent diffusion models opt for a two-stage approach. Initially, the autoencoder compresses training images, creating a perceptually equivalent, but computationally cheaper latent space for semantic training. The autoencoder finishes training prior to the second stage, where autoencoder weights are frozen and diffusion training is prioritized. 

### Autoencoder

Autoencoders have become ubiquitous across machine learning architecture. Their purpose is to receive an input and encode that input to a compressed representation with minimal information loss. The original LDM paper allowed for a pretrained autoencoder to accomplish this task. In practice, most implementations actually trained new autoencoders from scratch to adhere to the paper's loss functions. If you haven't already, I suggest you check out my page [explaining the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet_ML.md). It gives a broad overview of the typical encoder-decoder architecture. Autoencoders for LDMs offer more complexity than the U-Net, but maintain the same principles. 

Encoding to a latent space requires decisions on the size of the latent space. The authors experimented with multiple downsampling factors, ultimately determining that downsampling factors of 4 or 8 offered the best performance. Downsampling factors of 1 or 2 were considered prohibitively expensive, operating near pixel-space, and greatly slowing the training rate. Downsampling by a factor of 16 or more was determined to cause excessive information loss and low fidelity. We'll be focusing on downsampling by a factor of 4 which has more referrals in the original research paper than the factor of 8 model. We'll refer to this specific model as LDM-4 for the remainder of the page.

Make sure we emphasize that the role of the autoencoder is perceptual compression.

#### Encoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_encoder_architecture.png" alt="Illustration of LDM encoder" width="100%"
</p>

Constituting the encoder are a variety of convolution operations, ResNet blocks, attention operators, activation functions, and downsampling. [One Stable Diffusion implementation](https://github.com/CompVis/stable-diffusion/tree/main) is illustrated above. Through every step of the encoder operation, the aim is to efficiently condense image features. This is accomplished through an initial convolution, broadening the channels to 128, where pixel-space values are converted to feature embeddings. Immediately after this convolution are a pair of ResNet and attention blocks, providing propagation and self-attention of all image features. This continues to a loop of ResNet, attention blocks, and downsampling. In this loop, a pair of ResNet and attention blocks are applied twice to ensure feature propagation before downsampling. Downsampling reduces the height and width of the data by half, until we have compacted our data to the expected latent dimension size. Since each loop halves the height and width, we repeat the loop log<sub>2</sub>f times. For LDM-4, downsampling would be performed twice, LDM-8 would perform this loop three times. After exiting the loop, a sequence of ResNet-attention-ResNet blocks inspects the condensed image features before normalization stabilizes the data across all channels. A nonlinear activation function is applied elementwise through the Sigmoid function before another convolution operation controls our number of output channels. 

Talk about role of self-attention with image embeddings. Mention encoder's responsibility in finding perceptually-equivalent lower-dimensional space. First stage of autoencoder is perceptual compression, then comes semantic compression and learning in latent space.

#### KL-regularization

To stabilize the latent space and prevent any pockets of high variance, we apply a low-penalty KL-regularization scheme. [KL-regularization has been shown to be very effective](https://proceedings.neurips.cc/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf) in efficiently unifying diverging distributions. This serves to push the latent space to an approximation of a Gaussian distribution, smoothing out an otherwise unpredictable and high-variance encoding space. Unlike other encoder-decoder architectures, like the U-Net, the latent space for LDMs serves as more than an avenue to image reconstruction. The latent space of LDMs is a legitimate destination in its own right. All diffusion, generation, and denoising at inference time take place in this latent space; stabilizing this space affords a steadier venue for these operations. KL-regularization accomplishes this target, pushing the latent space distribution to an approximate Gaussian, and balancing our latent space.

The original paper also included VQ-GAN regularization, but ultimately KL was determined to provide better results. Include literature supporting this fact. Mention that the entire purpose of the regularizer is to foster a computationally-friendly latent space that is approximately Gaussian.

#### Decoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_decoder_architecture.png" alt="Illustration of LDM decoder" width="100%"
</p>

Our decoder is assembled from the same blocks as our encoder and performs similar functions. Every step of the decoder serves to reconstruct our latent encoding to its pixel-space reconstruction. In the training stage, these operations are focused on refining their weights to offer the most accurate mapping of a lower-dimensional encoding to the higher-dimensional image. Similar to the encoder, our first step is through a convolution block. Broadening the number of channels offers more environments to preserve information while upsampling. The broadened encoding is then passed through a ResNet-attention-ResNet sandwich, self-attending every encoded feature. The filtered representation is then passed through the same loop as our encoder, upsampling rather than downsampling. We upsample log<sub>2</sub>f times to expand our features to their original dimensions. For LDM-4, the features pass through the loop twice. Reaching pixel-space at the end of our loop, we normalize our image before applying the sigmoid function elementwise. Lastly, we pass it through a 3x3 convolution and arrive at our reconstructed image. 

Similar to the encoder, we perform fewer operations at the pixel-space dimensions. The intermediate stages have performed the necessary filtering of image features. Additional operations at higher dimensions could confuse our upsampling parameters and lead to a lower-fidelity reconstruction.

Also, initial ResNet in every loop is controlling number of channels. Also, touch again on importance of ResNet and attention blocks. DECODER HAS ONE EXTRA RESNET IN COMPARISON TO ENCODER!!!

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

The autoencoder solves the first stage of our training: perceptual compression. We've now arrived at a perceptually equivalent and computationally cheaper latent space for our second stage: semantic compression. Here, we'll learn the conceptual composition of images to ensure high fidelity for image synthesis. The semantic learning stage centers on three key ingredients: the scheduler, the U-Net, and the conditioner. Throughout the semantic training stage, the autoencoder is frozen to prevent any changes to its weights, and all learning takes place in the latent space.

Schedulers are algorithmic guides to the denoising process implemented through the U-Net architecture. Training revolves around learning the additive noise process to understand the guided reversal of noise in an image. Many scheduling algorithms have been developed over the years, but were thought to be inextricable from the model architecture until a [2022 paper by Song et. al](https://arxiv.org/pdf/2010.02502.pdf) suggested pre-trained models could utilize different schedulers at inference time within the same family of generative models, and [another paper by Karras et. al](https://arxiv.org/pdf/2206.00364.pdf) confirmed that scheduling algorithms could be entirely separated from the denoising architecture. Schedulers learn the Gaussian addition of noise to images to subsequently model the Gaussian removal of noise from images. The important parameters for these algorithms are: a linear or cosine schedule and a vector linked to the timestep of the iterative noise removal process. We can treat these parameters as a purely mathematical function guiding the U-Net's denoising of latents, thanks to the years of literature that have examined these algorithms for image synthesis performance. For more information on schedulers, I recommend reading the page I wrote focusing on [their literature and evolution](https://github.com/ejohansson13/concepts_explained/blob/main/Stable%20Diffusion/Schedulers_ML.md).

### U-Net

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/unet_architecture.png" alt="Screenshot of U-Net architecture taken from original research paper" width="40%"
</p>

If you're looking for information on the U-Net architecture, check out [my page](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet_ML.md) offering a brief summarization of its application within image segmentation. The U-Net is an encoder-decoder architecture popularized through its performance in computer vision tasks. In LDMs, the U-Net is responsible for the repetitive denoising of the latent. Ostensibly, the denoising architecture for LDMs does not have to be a U-Net, but [the U-Net's inductive bias for spatial data](https://arxiv.org/pdf/2105.05233.pdf) quickly led to it becoming ubiquitous for denoising in diffusion models. The U-Net functions architecturally similar to its image segmentation application, downsampling its input and expanding the number of channels to determine the conceptually critical image features before upsampling and condensing the number of channels to preserve spatially relevant information.

Within LDMs, U-Nets can be trained to either predict the denoised latent when passed a noisy latent or predict the amount of noise that needs to be removed from a latent to arrive at the denoised latent. Throughout training the U-Net is fed three inputs: the noisy latent, the magnitude of noise that was added to the latent, and the conditioning input. Since the perceptual training stage is complete, we can have confidence in the veracity of our encoded latents. We add a randomly sampled magnitude of noise to a clean, encoded latent to create our noisy latent. This magnitude equates to the timesteps of noise in the forward process we would take to arrive at our randomly sampled noise level. Since the added noise is Gaussian, and we have learned the mean and variance of the forward process from our scheduling algorithm, we can immediately scale up our added noise to our randomly sampled noise level. This noise is added to our clean latent and we pass in both the subsequent noisy latent and the noise level as inputs to the U-Net. More information on the conditioning and its interaction with the U-Net can be found [below](#conditioning). 

Knowing the number of timesteps of noise that was added to our latent, the U-Net begins iterating and progressively denoising the latent. Depending on its mode, either predicting the clean latent or the added noise, the U-Net will output either a latent or the predicted noise added to the latent. In either case, the output is compared to the ground truth via MSE. Throughout training, the U-Net learns how to progressively and iteratively denoise a latent. This becomes crucial at inference time where prompts will be fed in to the U-Net, accompanied by a randomly sampled latent, and an input number of timesteps in which to denoise the latent and synthesize an image. 

### Conditioning

Prompting our diffusion model requires the network's understanding of the prompt. We need an encoder to convert our prompt into embeddings which can interact with our latent embeddings in the U-Net. For the text-to-image case this would be a text encoder but, alternative encoders can deliver prompts through different media while remaining compatible with the LDM architecture. The original LDM paper utilized BERT, but more recent literature has demonstrated that the performance of an image synthesis model is strongly tied to the [performance of its text encoder](https://arxiv.org/pdf/2205.11487.pdf). For this reason, newer diffusion models [have opted for more powerful pre-trained text encoders](https://arxiv.org/pdf/2307.01952.pdf).

The conditioning plays a critical role in guiding the U-Net's denoising of the latent towards a definite destination at inference time. During training, we are afforded the luxury of a pre-determined destination with our provided clean latent. It's important to take advantage of that luxury so the network can correctly train its query, key, and value weight matrices for cross-attention. Latent embeddings serve as the query with textual embeddings providing the roles of key and value vectors. Performing cross-attention between these vectors emphasizes the relationship between the textual prompt and generated image. Cross-attention occurs at every layer for every timestep of the U-Net. Every token in the textual prompt has a thorough impact on the attention the network pays to the latent embeddings throughout the denoising process. By performing cross-attention at every layer, through every downsampling and upsampling operation, we ensure the uninterrupted propagation of information to every stage of the latent's progression through the network. The holistic training approach taken in the semantic training stage allows the U-Net to learn the successful denoising of latents while concurrently learning the conditioning impact on the ultimate latent destination.

## Inference

### Scheduler

Inference time schedulers are determined by pre-defined algorithms through previous literature. They are just initialized and applied to latent denoising to determine the amount of noise to be removed.

### U-Net

Pass in two of everything to U-Net for classifier-free guidance. Concatenated x (latent (sampled noise)) with itself, t (number of timesteps to remove noise) with itself, and conditioning c (empty prompt concatenated on to actual text prompt).

### Conditioning (Prompt)

Talk about the interpolation between the provided prompt and the guided denoising of the latent to ultimately result in a newly synthesized image.

Empty conditioning - classifier free guidance, passed through CLIP, used to condition U-Net. Classifier-free diffusion guidance greatly improves sample quality. 

### Decoder

# Future Steps

Already seen that increasing size of U-Net leads to improved results (SDXL). Can also add refiner/superresolution U-Net to upsample to more detailed image dimensions. LM performance can have significant effect on downstream image generation performance (Imagen). What other future directions can image synthesis take / what can be done to further improve performance?





# Citations

VGG16 architecture image - https://pub.towardsai.net/the-architecture-and-implementation-of-vgg-16-b050e5a5920b

Patch-based adversarial loss training image - Rao, S., Stutz, D., Schiele, B. (2020). Adversarial Training Against Location-Optimized Adversarial Patches. In: Bartoli, A., Fusiello, A. (eds) Computer Vision – ECCV 2020 Workshops. ECCV 2020. Lecture Notes in Computer Science(), vol 12539. Springer, Cham. https://doi.org/10.1007/978-3-030-68238-5_32
