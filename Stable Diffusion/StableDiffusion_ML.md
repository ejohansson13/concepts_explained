Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Similar to the rise of the transformer architecture for natural language processing tasks, [Latent Diffusion Models (LDMs)](https://arxiv.org/pdf/2112.10752.pdf) quickly grew in popularity for conditional image generation. Previous image synthesis architectures included [GANs](https://arxiv.org/pdf/1605.05396.pdf), [RNNs](https://arxiv.org/pdf/1502.04623.pdf), and [autoregressive modeling](https://arxiv.org/pdf/1906.00446.pdf) which [multiple](https://arxiv.org/pdf/2102.12092.pdf) [companies](https://arxiv.org/pdf/2206.10789.pdf) favored. However, the performance of diffusion models in zero-shot applications as well as their efficiency operating in the latent space, led to LDMs becoming the eminent architecture for a variety of generative tasks. Their compatibility with natural language, image, and audio inputs accelerated the multimodality of machine learning applications. On this page, you'll find an overview of the Stable Diffusion architecture, its training procedure, and inference steps.

We'll be focusing on the text-to-image case, but I urge you to check out the [original paper](https://arxiv.org/pdf/2112.10752.pdf) which described multiple other use cases, including image inpainting, conditioning on semantic maps, and layout-to-image, with image generation conditioned on annotated bounding boxes.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_diagram.png" alt="LDM diagram taken from original research paper" width="100%"
</p>

# Architecture

The latent diffusion network follows an encoder-decoder architecure interacting with a compressed, semantic latent space. Between the encoder and decoder blocks, there are three main stages: the scheduler, the denoising U-Net and the custom conditioner (again, we'll be focusing on the text conditioning case). The encoder is responsible for encoding images to their  latent representation. The U-Net is the vehicle carrying this latent representation through the latent space. The prompt suggests the latent space destination, and the scheduler guides the U-Net through the latent space to the preferred destination. This destination, decided by the U-Net, scheduler, and prompt, is the closest latent space representation to the final image. The destination latent is then decoded and transformed to a pixel-space image adhering to the details described in the provided prompt.

Each of these stages will be covered in more detail below, and I've split the architecture details in two. [Training](#training) covers each stage's role throughout the training process. [Inference](#inference) offers the same details for the inference process, describing how user input is synthesized to a 512x512 pixel image.

## Training

Stable Diffusion models are trained on a wide variety of images pulled from the web. A large dataset of images are pumped in for the network to learn a variety of visual themes and configurations. These images are taken apart through a forward process consisting of the time-controlled, sequential addition of Gaussian noise to a training image. This iterative destruction of images teaches the network the probability distribution of additive noise to the point of incoherence in an image. It also teaches the network the reverse probability distribution of removing Gaussian noise to generate a coherent image from random noise. The process of generating lucid images from noise is referred to as the reverse diffusion process. Importantly, LDMs were not the first network architecture to operate under the diffusion theory. The idea that models can learn to rebuild images from noise by decomposing training images can be found as early as [2015](https://arxiv.org/pdf/1503.03585.pdf). However, previous architectures operated on the pixel-space of images, reserving their utility for entities who could accomodate the resource-intensive training process. The distinction of latent diffusion models is autological. Rather than operate on the pixel-space, they first compact each image representation, encoding them to a latent space.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_graph.png" alt="Graph demonstrating perceptual vs semantic compression training tradeoff taken from 2022 paper on latent diffusion models" width="50%">
</p>

The above graph demonstrates that the majority of bits constituting a digital image correspond to high-frequency, perceptual details. In contrast, relatively few bits comprise the semantic information of the image. Unlike previous diffusion models that tried to [jointly balance perceptual and semantic training loss terms](https://arxiv.org/pdf/2106.05931.pdf), latent diffusion models opt for a two-stage approach. Initially, the autoencoder compresses training images, creating a perceptually equivalent, but computationally cheaper latent space for semantic training. Learning the diffusion process is prioritized in the second stage of training. Semantic training freezes the autoencoder weights and prioritizes diffusion training, learning the reconstruction of coherent images from random white noise. 

### Autoencoder

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/autoencoder_architecture.png" width="70%"></img>
</p>
<p align="center">
  <em>Image taken from Lilian Weng blog: https://lilianweng.github.io/posts/2018-08-12-vae/</em>
</p>

Autoencoders are a staple of machine learning architectures. Their purpose is to encode and downsample an input to a compressed representation before decoding and upsampling the input to its original dimensionality with minimal information loss. Autoencoders are the first and last stage of Latent Diffusion Models. Their role is to create a stable, learnable latent space containing the distribution of compressed images from which future images can be generated. This space must contain all relevant semantic image information while obscuring superficial perceptual details. The latent space of the autoencoder is perceptually equivalent to the pixel-space of images, while offering computationally friendlier calculations owing to its lower-dimensionality scope.

Encoders perform this initial compression role, with decoders carrying the opposing responsibility. Decoders need to successfully rescale lower-dimensional latents to pixel-space images. Their weights need to be carefully calibrated to avoid the reintroduction of errors when scaling a latent to a higher-dimensional representation. Jeopardizing the fidelity of the image when transitioning to higher dimensions would render latent-dimension modeling obsolete.

Encoding to a latent space requires decisions on the size of the latent space. The authors experimented with multiple downsampling factors, ultimately determining that downsampling factors of 4 or 8 offered the best performance. Downsampling factors of 1 or 2 were prohibitively expensive, operating near pixel-space, and slowed the training rate. Downsampling by a factor of 16 or more led to excessive information loss and low fidelity. Compression at that scale cannibalized the semantic information present in the training data. We'll be focusing on downsampling by a factor of 8, which was used to generate the image examples in the paper. We'll refer to a model of this downsampling factor as LDM-8 for the remainder of the page.

##### ResNet blocks
Before explaining the encoder and decoder aspects of the variational auto-encoder, covering their building block is conceptually relevant. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385), released in 2015, introduced the concept of a residual network for improved performance in image processing tasks. At the time, conventional wisdom held that network depth was inexorably tied to model performance. In reality, unstable gradients and training accuracy degradation highlighted the flaw in arbitrarily deep models. Adding layers and layers in the hopes of modeling an implicit underlying mapping is inferior to explicitly fitting a referenced residual mapping. This was the argument made in the ResNet paper. The authors demonstrated that allowing shortcuts between layers of the neural network allowed the network to determine the relevancy and necessity of each layer. Layers deemed unnecessary to the model's decision-making could be mapped to an identity function, allowing the network's signal to propagate unperturbed.

ResNet blocks are residual blocks. Offering a shortcut connection between the input and the propagated signal, they limit each block's responsibility to incremental, residual signal changes. The utility of these blocks and their shortcut connections allow their application in networks of arbitrary depth. Designating two paths for the signal allows the network to determine the relative significance of each path, and weight the correct decision-making path prior to their aggregation. Improved decision-making is compounded by backpropagation. Backpropagation with residual blocks quickly allows larger gradients to flow to earlier layers in the network via the architectural shortcut of the original signal. Convolutional kernel weights are dictated by backpropagation, allowing the network to determine the significance of each kernel's contributions. Weights that are deemed unhelpful or unnecessary can be minimized, encouraging the original input to propagate, creating an identity mapping.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/residual_block_types.png" width="60%"
</p>
<p align="center">
  <em>Image from [2]</em>
</p>

These blocks can have multiple compositions. A [follow-up paper in 2016](https://arxiv.org/pdf/1603.05027) by the same authors examined these compositions and their relative performance. Ultimately they determined the full pre-activation option (e in the above diagram) offered the best performance. This conclusion was supported by two findings. Identity mapping as the shortcut for the original signal encourages optimization, promoting backpropagation to earlier layers, as described above. This accelerates training in the early stages, as the network quickly adjusts its weights in accordance with feedback. Any risk of overfitting is mitigated by their second finding: normalization in the pre-activation path prevents overfitting. Consistent regularization of the propagated signal prevents overt weight adoption of the training data distribution, allowing for successful generalization to novel data. The above configuration is utilized in the LDM autoencoder as well, illustrated below. 

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/ResNet_composition.png" width="80%"
</p>

ResNet blocks' success with arbitrarily deep networks defines their success with autoencoders, where network depth is decided by the desired latent space dimensionality. Their ability to combat overfitting and generalize to new data distributions projects well to image compression, where the data distribution encompasses all pixel-space visual configurations. ResNet blocks extract and preserve significant image features with minimal information loss throughout the autoencoder.

#### Encoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_encoder_architecture.png" alt="Illustration of LDM encoder" width="100%"
</p>

Constituting the encoder are a mix of ResNet blocks, downsampling, convolution, attention blocks, normalization, and activation functions. [One Stable Diffusion encoder implementation](https://github.com/CompVis/stable-diffusion/tree/main) is illustrated above. Through every step of the encoding operation, the aim is efficient consolidation of image features. Entering the encoder, the image input is convolved with a 3x3 kernel. This convolution broadens the pixel-space RGB image to 128 channels, where pixel-space values are converted to feature embeddings.

Following convolution, a stage of sequential ResNet blocks preceding a downsampling operation is visible. This coalition accomplishes the model’s downsampling objective. The encoder can repeat the progression of image features through this stage until the feature embedding dimensions satisfy the latent space dimensionality. ResNet blocks serve as feature extractors, determining the important semantic information of the input image. Their propagation of semantically significant image features abstracts superficial perceptual details. Removing high-frequency details from the image composition in concert with downsampling ensures that all data of latent space dimensionality is semantically equivalent to their higher dimensionality counterparts. Each downsampling operation halves the height and width of their inputs. Compression of half of the data at each downsampling operation requires the downsampling factor to be a power of 2. For LDM-8, data is downsampled three times before exiting the loop and progressing along the encoding path. 

Outside of the downsampling loop, image features are propagated through three additional ResNet blocks. These blocks continue the data progression, preserving and stabilizing image features throughout training. Following these blocks, the training data arrives at an attention block. Throughout the encoder architecture, feature analysis and understanding has been performed by convolution. Convolution is superb in the detection of proximal image features. Holistic understanding of the image can require a different approach.  Similar to self-attention in text, applying self-attention to image features enables the understanding of longer interdependencies between features and broadens the network's understanding of the overall image. The comprehensive understanding of image features provided by self-attention is especially important at lower dimensions where the compression of perceptual information raises the risk of information loss. Applying self-attention in the encoder, at the latent space dimensionality, mitigates that risk. After the attention block, another ResNet block is applied for stabilization and propagation of the learned image features. Exiting the downsampling loop and achieving latent space dimensionality, the three sequential ResNet blocks, attention block, and final ResNet block are focused on preserving and stabilizing the learned image features in their compact dimensionality.

Having abstracted perceptual details and arrived at a compact, semantic representation of the image data, features are passed through a normalization function (for further stabilization) and an activation function (for further nonlinear data expression) before arriving at a final convolution operation. This convolution determines the number of relevant channels within the latent space, preparing the data both for regularization and semantic training compatibility.

#### KL-regularization

To stabilize the latent space and prevent any pockets of high variance, a low-penalty KL-regularization scheme is applied. [KL-regularization has been shown to be very effective](https://proceedings.neurips.cc/paper/2020/file/8e2c381d4dd04f1c55093f22c59c3a08-Paper.pdf) in unifying diverging distributions. This serves to push the latent space to an approximation of a Gaussian distribution, smoothing out an otherwise unpredictable and high-variance encoding space. Unlike other encoder-decoder architectures, the latent space for LDMs serve as more than an avenue to image reconstruction. The latent space of LDMs are a destination. All diffusion, generation, and denoising at inference time take place in this latent space; stabilizing this space affords a steadier venue for these operations. KL-regularization accomplishes this target, pushing the latent space distribution to an approximate Gaussian. Approximating a Gaussian distribution becomes important at inference time. Removing Gaussian noise from an approximately Gaussian distribution accelerates an iterative process, expediting the denoising process.

The original paper also considered VQ regularization but, Table 8 [in the original LDM paper](https://arxiv.org/pdf/2112.10752) demonstrated superior empirical performance for the KL-regularized LDMs when compared to their VQ-regularized counterparts. VQ-regularization was built on the theory of the [Vector Quantised-Variational AutoEncoder](https://arxiv.org/pdf/1711.00937) where the latent space of the autoencoder was considered as a discretized code-book of latent vectors.

Training in the latent space of a KL-regularized model requires awareness of the latent space variance. Mirroring a Gaussian distribution eases the generative process but, introduces a higher signal-to-noise ratio tied to the latent space variance, which can then effect image quality. Rescaling by the [component-wise standard deviation of the latents](https://arxiv.org/pdf/2112.10752) decreases the corresponding signal-to-noise ratio and restabilizes the latent space to a unit variance. VQ-regularization assumes a uniform prior distribution with a variance of 1, and does not require rescaling.

#### Decoder
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/LDM_decoder_architecture.png" alt="Illustration of LDM decoder" width="100%"
</p>

The decoder is responsible for reconstructing the latent image representation to a pixel-space image with minimal information loss. Concurrently, it is responsible for reintroducing the high-frequency, perceptual details abstracted away by the encoder. [One implementation](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py) is depicted above. The decoder complements the encoder and starts with a similar sequence of operations to the end of the encoder path. Latents are broadened from their minimal dimensionality to image features through an initial convolution. This operation broadens the number of channels accompanying the image features for compatibility with the decoder path. Following the convolution, image features progress through an attention block bookended by ResNet blocks. Repeating their functionality from the encoder path, these blocks offer proximal and holistic perspectives on the compressed image features, preserving the learned semantic image information.

Exiting the top row of operations, image features enter an upsampling loop to rescale image dimensions. These three ResNet blocks in conjunction with the upsampling operation control the height and width of the data. Similar to the encoder, this loop is variable and can be enforced depending on the required rescaling of the data. For LDM-8, this loop would be run three times (identical to the encoder). Examining the encoder and decoder, the distinction in number of ResNet blocks in each respective loop becomes apparent. The additional ResNet block when decoding is necessary due to the heightened responsibility of the decoder at inference time. The encoder has no role at inference time, while the decoder is responsible for translating a randomly sampled latent to a realistic, coherent image. The purpose of the encoder was to create a perceptually equivalent latent space. The role of the decoder is the seamless reintroduction of those high-frequency visual details, affirming that latents and their decoded pixel-space counterparts are perceptually equivalent. For that reason, the decoder is allotted additional parameters. Again, the versatility of ResNet blocks is beneficial. The decoder is designated additional parameters, but their utilization is at the network’s discretion. If unnecessary, these parameters can instead model the identity function with no detriment to network performance. 

Following the upsampling loop, another three ResNet blocks are encountered, continuing the stabilization and propagation of high-dimension image features. They are further normalized and passed through an element-wise activation function before being convolved to arrive at a pixel-space image with three channels for RGB. 

Architectural improvements are constantly discovered. The analysis focused on in the preceding sections was released in 2022, accompanying the research paper. Since then, Stable Diffusion has been released with a 2.0, 2.1, and XL version. Google released Imagen. OpenAI released Dall-E 3. However, the role of autoencoders in latent diffusion models hasn’t changed. Their fundamental responsibility is the compression of image information to a computationally cheaper latent space. The description provided above is one possible avenue to achieving that goal, but not the only one.

#### Metric

Two metrics are used to determine the autoencoder's success: perceptual loss and patch-based adversarial loss. 

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/vgg16_architecture.png" alt="Illustration of a VGG16 convolutional neural network" width="55%"
</p>
  
Perceptual loss, or [LPIPS](https://arxiv.org/pdf/1801.03924),  measures the semantic understanding of the reconstructed image in comparison to the original. Both the original and reconstructed images are passed through a pre-trained [VGG16](https://arxiv.org/pdf/1409.1556) convolutional neural network. Following the completion of a convolutional layer (sequence of blue rectangles in above diagram, immediately prior to max pooling), the original and reconstructed image's outputs are compared. Each convolutional layer is terminated with a ReLU function, and each image's features are compared here, prior to the subsequent max pooling operation, via mean-squared error (MSE). Five of these locations exist in the VGG16 architecture. The total MSE is summed across the five output locations to determine the perceptual loss of the reconstructed image. MSE in the feature-space is preferred to typical Euclidean-space losses, such as L1 or L2 loss, which depend on pixel-wise comparison. Minimizing the Euclidean distance between two images assumes pixel-wise independence and averages proximal pixel differences between images, encouraging blurring. The success of the perceptual loss metric can be [dependent on the network](https://arxiv.org/pdf/2302.04032.pdf) employed for semantic comparisons, and [later literature](https://arxiv.org/pdf/2307.01952.pdf) would demonstrate a decreased emphasis on perceptual loss.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/patch_based_adversarial_loss.jpeg" alt="Example of patch-based adversarial loss" width="40%"
</p>

[Patch-based adversarial loss](https://arxiv.org/pdf/1611.07004.pdf) borrows from GAN theory, introducing controlled patches of noise to reconstructed images while training a discriminator to detect the noisy patches. Introducing localized patches [enforces pixel-space realism](https://arxiv.org/pdf/2012.09841.pdf). Aiding a discriminator in the detection of generated images by introducing scalar patches of noise encourages the decoder to maintain perceptually important high-frequency details while decoding from the latent space. The patch-based loss is not utilized for an initial chunk of training (50k or so steps), allowing the autoencoder to establish robustness in its encoding-decoding paradigm. Immediately training with both the perceptual and adversarial losses would lead to an overly powerful discriminator and a weaker autoencoder. 

Perceptual loss is focused on wholly improving the autoencoder. Parameters for both the encoder and decoder halves are tuned in response to the success of the reconstructed image in mimicking the semantic information of the original image. Patch-based adversarial loss is far more attentive to optimizing the decoder. Successful abstraction of superficial image details leads to a computationally cheaper latent space and more efficient image generation. It also runs the risk of generating bleak or somber images. Maintaining realism requires fixation on the successful reintroduction of high-frequency details at the decoding stage. This is the intuition behind patch-based adversarial loss, and why both loss functions were implemented to train the autoencoder.

##### ResNet blocks
The only addition to the ResNet blocks utilized in the diffusion architecture compared to those employed in the autoencoder is the temporal factor. The iterative denoising foundation of diffusion requires a monitor for the temporal progression. Throughout the diffusion process, iterations are measured in timesteps. Integrating those timesteps into the diffusion process means the timestep data has to be everpresent in the diffusion architecture. A diagram for the diffusion ResNets is provided below. The only change is the timestep inclusion where the timestep data, presented as a matrix, enters a Linear function to become dimensionally compatible with the denoising latent.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/SD_Images/ResNet_diffusion_composition.png" width="40%"
</p>

### Scheduler

The autoencoder solves the first stage of our training: perceptual compression. We've now arrived at a perceptually equivalent and computationally cheaper latent space for our second stage: semantic compression. Here, we'll learn the conceptual composition of images to ensure high fidelity for image synthesis. The semantic learning stage centers on three core components: the scheduler, the U-Net, and the conditioner. Throughout the semantic training stage, the autoencoder is frozen to prevent any changes to its weights, and all learning takes place in the latent space.

Schedulers are algorithmic guides to the denoising process implemented through the U-Net architecture. Training revolves around learning the additive noise process to understand the guided reversal of noise in an image. Many scheduling algorithms have been developed over the years, but were thought to be inextricable from the model architecture until a [2022 paper by Song et. al](https://arxiv.org/pdf/2010.02502.pdf) suggested pre-trained models could utilize different schedulers at inference time within the same family of generative models and, [another paper by Karras et. al](https://arxiv.org/pdf/2206.00364.pdf) confirmed that scheduling algorithms could be entirely separated from the denoising architecture. Schedulers learn the schedule of Gaussian noise addition to images to subsequently model the removal of Gaussian noise from images. The important parameters for these algorithms are: a linear or cosine schedule and a vector linked to the timestep of the iterative noise removal process. These parameters can be understood as mathematical functions guiding the U-Net's denoising of latents, thanks to the years of literature supporting them. For more information on schedulers, I recommend reading the page I wrote focusing on [their literature and evolution](https://github.com/ejohansson13/concepts_explained/blob/main/Stable%20Diffusion/Schedulers_ML.md).

Training: telling U-Net how many timesteps of noise to add (# of timesteps is randomly generated), we add that many timesteps of noise at once (we know mean and variance from scheduling algorithm). We then have the U-Net predict how much noise should be removed at each timestep, compare prediction to ground-truth via L2 norm.

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

Classifier-free guidance.

## Inference

At inference time, the LDM is applying its learned weights through progressive denoising, conditioning, and decoding to align randomly sampled noise with a user-provided natural language text prompt. The end product is expected to be a visually coherent generated image with high-frequency details. Inferring the product begins in the latent space. As mentioned above, Gaussian noise is randomly sampled to form our latent. If the perceptual compression training stage was successful, we will have arrived at an approximately Gaussian distribution in our latent space. This simplifies the sampling process and allows for easy access to a noisy latent expected to follow Gaussian properties. We treat this latent equivalently to noisy latents utilized in the semantic compression training stage. Treating the latent as a meaningfully compressed image with added noise, we begin the denoising process.

### Diffusion

The noisy latent is iteratively fed through the U-Net in conjunction with the provided prompt, progressively removing chunks of noise at every step. Dependent on the sampler schedule, this process may be deterministic or stochastic. From the training process, the U-Net is expected to successfully remove predictable amounts of noise in accordance to the provided timestep. During inference, we provide the U-Net with sequential chunks of timesteps such that the U-Net removes noise proportional to those chunks during each iteration. Given a randomly sampled noisy latent, the sequential denoising process could theoretically lead to any non-deterministic output. By providing conditioning during the semantic training process, the U-Net learns both the successful denoising of a latent and the navigation to a destination latent provided from a conditioning input. The navigation through latent space provided by our conditioning input is finely tuned through the training of cross-attention weight matrices in the diffusion training section of our LDM. **Something about classifier-free guidance** 

Reiterate relationship between regularized Gaussian latent space and ease of denoising.

Iterative denoising of the latent to the provided number of inference steps work in concert with the user's provided conditioning to denoise the latent and navigate it to the preferred destination. The success in the inference stage will be dependent on the success of the training stage and complexity of the selected prompt. Models asked to generate images similar to their training data will perform better. Larger models with more parameters will often perform better at zero-shot tasks. Comprehensive language understanding tied to the performance of the conditioner can also have a significant impact on the success of the model at inference time.

#### Scheduler

Inference time schedulers are determined by pre-defined algorithms through previous literature. They are just initialized and applied to latent denoising to determine the amount of noise to be removed.

#### U-Net

Pass in two of everything to U-Net for classifier-free guidance. Concatenated x (latent (sampled noise)) with itself, t (number of timesteps to remove noise) with itself, and conditioning c (empty prompt concatenated on to actual text prompt).

#### Conditioning (Prompt)

Talk about the interpolation between the provided prompt and the guided denoising of the latent to ultimately result in a newly synthesized image.

Empty conditioning - classifier free guidance, passed through CLIP, used to condition U-Net. Classifier-free diffusion guidance greatly improves sample quality. 

### Decoder

Denoising our latent to the suggested latent destination is the first stage of our inference process. Arriving at a successful pixel-space image from our latent is still dependent on the decoder. Only the decoding portion of our autoencoder is employed at inference time. Ideally, it performs successfully scaling the denoised latent to higher dimensions and translating the image's features to pixel values. **something about perceptual loss** The role of the patch-based adversarial loss in the training stage encourages the decoder to ensure fidelity across the image and promotes the inclusion of high-frequency details in the final decoded product. Training a discriminator in partnership with the autoencoder pushes the decoder to ensure decoded images are as realistic and inclusive of superficial details as the real images to which they are compared. Luckily, this push towards realism and detailed images in training doubles as an insurance policy for the fidelity of generated images at inference time.



# References
[1] LDM paper - https://arxiv.org/pdf/2112.10752

[2] SDXL paper - https://arxiv.org/pdf/2307.01952

[3] ResNet paper - https://arxiv.org/pdf/1512.03385

[4] Improved ResNets - https://arxiv.org/pdf/1603.05027

[5] VGG16 architecture image - https://pub.towardsai.net/the-architecture-and-implementation-of-vgg-16-b050e5a5920b

[6] Patch-based adversarial loss training image - Rao, S., Stutz, D., Schiele, B. (2020). Adversarial Training Against Location-Optimized Adversarial Patches. In: Bartoli, A., Fusiello, A. (eds) Computer Vision – ECCV 2020 Workshops. ECCV 2020. Lecture Notes in Computer Science(), vol 12539. Springer, Cham. https://doi.org/10.1007/978-3-030-68238-5_32

[7] Lilian Weng blog on LDMs - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

[8] Aleksa Gordic video on LDMs - https://www.youtube.com/watch?v=f6PtJKdey8E

[9] Umar Jamil video on LDMs - https://www.youtube.com/watch?v=ZBKpAp_6TGI
