![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)

# U-Net

The original U-Net research paper was released in 2015 and demonstrated improved success with biomedical image segmentation. The architecture's efficiency and performance with a small dataset announced its efficacy as a tool for computer vision tasks. In this page, we'll discuss the U-Net and its architecture. This page assumes familiarity with machine learning technology and operations. If you don't have any machine learning experience, check out my [other page on the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet.md), which makes no assumptions of any machine learning background.

## Architecture

The U-Net is a fully convolutional neural network following an encoder-decoder architecture, deriving its name from the signature U-shape of its encoder and decoder paths. The encoder path represents the descent of the U and downsamples the image, condensing it to its most important features. This compact representation, preserving image features, is then upsampled and ascends the network, with the output image replicating the original image's dimensions. Skip connections stretch across the symmetry of the network, providing additional context for upsampling. This added context, connecting equivalent stages in the architecture across paths, boosts the image reconstruction accuracy. Below, we'll take a look at each section in a little more detail.

### Downsampling

<p align="center" width="100%">
  <img src="/UNet/Images/first_downsampling_step.png" alt="The first max pooling operation performed on the contracting path of the U-Net" width="20%"
</p>

We can consider the encoder path as consisting of a sequence of stages. These stages have the same composition: convolution with a 3x3 kernel of stride 1, ReLU activation function applied elementwise, another convolution mimicking the first, another ReLU function, then a 2x2 max pooling operation with stride 2 for image downsampling. This sequence continues until we arrive at the bridge marking the end of the encoding path. With every stage, we halve the image dimension's while doubling the number of channels. Duplicating the channels of an image offers additional perspectives on the features advanced by the network. From another perspective, we trust the network to select the most important image features. We then double the observational environments for the network to monitor these features.

The U-Net architecture is predicated on two core tenets: efficiency and efficacy. The continuous contraction of our image dimensions reduces the necessary spatial dimensions for our convolutional calculations. This compact encoding also ensures that only the most important features survive each stage of the encoding path. By steadily compressing the image features, our network provides a cheaper computation space for its framework.

### Bridge

<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="35%"
</p>

The bridge, or bottleneck, of our architecture serves a nominal responsibility. Connecting the encoder and decoder stages of the network, it performs the same convolution and activation function sequence as encountered in the encoding stage. One of two connection structures between the encoding and decoding paths, it reduces our image to its most concentrated representation before passing the features to the expanding path.

For custom architectural purposes, the location of the bridge offers an exercise in deliberation. Should you keep four encoding stages before reaching the bridge? Should the image's smallest dimensions be 32x32, smaller, or larger? Depending on the size of your training images, you may want to mirror the bridge's positioning to the original research paper (they utilized original dimensions of 512x512). When training with larger images, you could be concerned that contraction to that degree will lead to the obfuscation of image features. If training with smaller images, you may not be giving the network an opportunity to filter unnecessary information from the image. Ostensibly the bridge only serves to connect the contracting and expansive paths, but it serves as a reminder: the devil is in the details.

### Skip connections

<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="75%"
</p>

Skip connections offer another framework for linking the encoder and decoder. Unlike the bridge, it operates on respective steps, connecting corresponding stages symmetrically across the architecture. After passing through an encoding stage (prior to being downsampled), image features are transmitted and concatenated on to the equivalent decoding stage. The original research paper accomplished this by cropping the encoder stage image to satisfy the dimensions of the decoder stage image. Newer implementations avoid this extra step by implementing padding in all convolutional operations. 

Skip connections concatenate decoder features with saved copies of encoded features. Each stage in the architecture may focus on different feature attributes. Decoded features often contain the image's semantic information, while encoded features will highlight spatial information. By concatenating the representations atop one another, the network benefits from both contexts. Skip connections also provide another opportunity for features that were previously discarded by the network. Reproducing discarded information offers another opportunity for the network to determine the feature's merits. Details initially considered uninformative can still benefit the network's image understanding.

### Upsampling

<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="40%"
</p>

Decoding our encoded features involves a similar process to the contracting path, in reverse. We have encoded our image to its most important features and now require their expansion to resemble our original image dimensions. To decode our image representation, we perform transpose convolutions to upsample our representation before passing the features through another block of convolution and activation operations. The specifications for these operations are equivalent to the encoding blocks (3x3 convolution, ReLU, repeat). At every decoding block, we are sweeping a magnifying glass over the image, highlighting and preserving the most important features for the final output.

Compressing our image to a compact representation emphasizes the most important image features while allowing an efficient computational space. For a functioning model, we need to reassemble these features for an understandable output without losing information in the process. The expanding path needs to efficiently reassamble our image from smaller dimensions while maintaining the spatial consistency of higher dimensions. The decoder reassembles our images, augmenting the features at every stage via skip connections and the multiple contexts provided through the image's channels. It consolidates the information present from every channel, skip connection, and upsampled feature for optimal performance in its specific computer vision task.

#### Final Layer

After having trained on its myriad of images, the network employs a 1x1 convolutional layer at the end of the architecture to control the number of output channels. This final layer is ultimately responsible for the image's computer vision task success.

## Impact
- image segmentation

- image classification

- image synthesis

- image restoration

- image superresolution
