![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)

# U-Net

The original U-Net research paper was released in 2015 and demonstrated SOTA performance with biomedical image segmentation. The architecture's success with a small dataset announced its efficacy for computer vision tasks. Designed for localization, the symmetric design shared feature channels across the contracting and expansive paths. The amalgamation of upsampled features with their contracting path complements produced additional context as the network propagated image features and determined labels for individual pixels in the image.

This page will discuss the U-Net and its architecture. It assumes familiarity with machine learning technology and operations. If you don't have any machine learning experience, check out my [other page on the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet.md), which makes no assumptions of a machine learning background.

## Architecture

The U-Net is a convolutional neural network following an encoder-decoder architecture. It derives its name from the signature U-shape of its encoder and decoder paths. The encoder path represents the descent of the U and downsamples the image, condensing the image information to its most important features. The compacted representation, preserving image features, is then upsampled and ascends the network, with the output image matching the original image's dimensions. Skip connections stretch across the symmetry of the network, providing additional context for upsampling. Augmenting the decoding path with context from the corresponding stage in the encoding path boosts the image reconstruction accuracy. Below, we'll take a look at each section in a little more detail.

### Convolution Block

The U-Net architecture is a sequence of blocks responsible for carrying out the primary function of the network: feature analysis. These blocks have the same composition: convolution with a 3x3 kernel, the ReLU function applied elementwise, a second 3x3 convolution, and a second ReLU function. In the original paper, every convolution operation utilized a 3x3 kernel, a stride of 1, and no padding. Newer variations of the U-Net have opted for padding to prevent dimensionality loss.

Convolution analyzes windows of the image at a time. Propagation of image features through these blocks ensures local feature analysis through convolution. The repeated applications of convolution act as decision-makers weighting the most important image features. Applying activation functions directly following convolution introduces nonlinearity to accurately model the complex relationships in visual data. Constructing the U-Net architecture with these blocks ensures consistent, localized feature analysis and complex modeling of propagated features.

### Contracting Path (Encoder)

<p align="center" width="100%">
  <img src="/UNet/Images/first_downsampling_step.png" alt="The first max pooling operation performed on the contracting path of the U-Net" width="20%"
</p>

In the encoding path, the described blocks (depicted as blue arrows in the above diagram) are applied twice before changing the feature dimensions. This is consistent throughout the contracting path: image features are passed through two blocks of convolution and activation function operations before being downsampled. The first block is also responsible for doubling the number of channels of the image features. As the image features are continuously contracted, doubling the number of channels mitigates the risk of information loss, allowing the network additional avenues to analyze the image data.

Each downsampling operation (red arrow in the above diagram) is a 2x2 max pooling operation with a stride of 2, taking the maximum value in a 2x2 window of the image features. Max pooling is an effortless downsampling application, halving the height and width of the image features. Repeated applications quickly distill the high dimensional image to its critical features.

### Bridge

<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="35%"
</p>

After repeated downsampling, image features arrive at the bridge, illustrated above. The bridge, or bottleneck, of the architecture serves a nominal responsibility. Connecting the encoder and decoder stages of the network, it performs the same convolution and activation function sequence as encountered in the encoding stage. One of two connection structures between the encoding and decoding paths, it reduces image features to their lowest dimensionality before advancing features to the expanding path.

For custom architectures, the location of the bridge in the architecture is a deliberate choice. Determining the lowest dimensionality for image features requires analysis of the risk of information loss at lower dimensions, the number of layers provided to the network to analyze image features, and the number of model parameters compatible with your available compute resources.

### Skip connections

<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="75%"
</p>

Skip connections (gray arrow in the above diagram) offer another framework for linking the encoder and decoder, connecting corresponding stages symmetrically across the architecture. After passing through an encoding stage (prior to being downsampled), image features are transmitted across the architecture and concatenated onto the equivalent decoding stage. The original research paper accomplished this by cropping the encoder stage image features to satisfy the dimensions of the decoder stage features. Implementing padding for convolution eliminates this requirement.

Skip connections concatenate decoder features with their mirrored encoded features. Decoded features often contain the image's semantic information, while encoded features will highlight spatial information. By concatenating these representations with each other, the network benefits from both contexts. A simplified visualization of this can be seen below, taken from [this video](https://www.youtube.com/watch?v=NhdzGfB1q74) explaining the U-Net architecture.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

Augmenting the semantic decoder-stage image features with their spatially aware encoder-stage counterparts allows the network to fixate on the physical position of image features, contributing to the U-Net’s groundbreaking success on computer vision tasks.

### Upsampling

<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="40%"
</p>

Decoding the encoded features involves a similar process to the encoding path, in reverse. After distilling the image to its most important features, the image now needs to be reconstructed to its original dimensions. The encoding path employs downsampling for control of the image feature dimensions, and the decoding path employs upsampling (illustrated as the green arrow above) to rescale image information.

Upsampling was implemented via “up-convolution”. Every feature value was quadrupled, doubling the number of rows and columns of feature data before being passed through a 2x2 convolution to halve the number of channels. Halving the number of channels creates space for the image features arriving from the skip connection. In the image above, the number of channels from the decoder stage features are convolved down to 64 channels following the upsampling operation (blue half of rectangle above green arrow). Augmenting the decoder stage features are the encoder stage features arriving via skip connection (white half of rectangle above green arrow). Concatenating the two feature stages together, the network arrives at the 392x392x128 dimensionality of image feature representation depicted above. 

After upsampling, image features are propagated through the same blocks of convolution and activation functions [covered above](#convolution-block) to further refine the semantic image information of the image. Additionally, the encoder stage features arriving from the skip connection are incorporated into the decoder stage features, providing additional context to the network as it advances features.

#### Final Layer

After training, the network employs a 1x1 convolutional layer at the end of the architecture to align the number of output channels for comparison to the ground-truth data. Each feature vector in the training data is mapped to the desired number of classes.

### Loss Metrics

The pixel-space image containing the preferred number of output channels is compared with the ground-truth masks. Cross-entropy quantifies the distance between the model's predicted output and the ground-truth. Following cross entropy, the determined loss is backpropagated through the network, fine-tuning network parameters and improving future predictions. Over enough iterations, the model converges on the ground-truth images and becomes a competent performer of the computer vision task.

## Impact

The U-Net's success with image processing tasks led to its ubiquity for computer vision problems. Similar to the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) architecture's triumph with natural language processing tasks, the U-Net quickly found success with a variety of computer vision tasks, including: image segmentation, image synthesis, and image superresolution. A significant component of the U-Net's success with image processing tasks arrives from its [inductive bias](https://arxiv.org/pdf/2105.05233). The incorporation of spatial information from encoder-stage features by the decoding stage ensures accurate pixel-based positioning of all objects in the reconstructed image. This compatibility with high-resolution data is particularly useful for image synthesis. Following the success of [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239), U-Nets became the signature backbones of diffusion models and continue to be employed in the most competitive [contemporary image generation architectures](https://cdn.openai.com/papers/dall-e-3.pdf).

Thanks to Olaf Ronneberger, Philipp Fischer, and Thomas Brox, the arrival of the U-Net accelerated performance in computer vision tasks and contributed heavily to the advancement of diffusion models, one of the most exciting applications of generative artificial intelligence in the last five years.
