![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)

The original U-Net research paper was released in 2015 and demonstrated improved success with biomedical image segmentation. The architecture's efficiency and performance with a small dataset announced its efficacy as a tool for computer vision tasks. In this page, we'll discuss the U-Net, its architecture, training process, and various implementations. This page assumes familiarity with machine learning technology and operations. If you don't have any machine learning experience, check out my [other page on the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet.md), which makes no assumptions of machine learning experience.

## Architecture

The U-Net is an encoder-decoder architecture, deriving its name from the signature U-shape of its encoder and decoder paths. The encoder path represents the descent of the U and downsamples the image, condensing it to its most important features. This compact representation, preserving image features, is then upsampled and ascends the network, with the output image replicating the original image's dimensions. Skip connections stretch across the symmetry of the network, providing additional context for upsampling. This added context, connecting equivalent stages in the architecture across paths, boosts the image reconstruction accuracy.

### Downsampling

We can consider the encoder path as consisting of a sequence of blocks. These blocks have the same composition: a convolution with a 3x3 kernel of stride 1 and 0 padding, followed by an elementwise ReLU activation function, another convolution layer with the same parameters as the first, another ReLU function, then a 2x2 max pooling operation with stride 2 to downsample the image for the next operation block. This sequence continues until we arrive at the end of the encoding path at the bottleneck. At each convolutional operation, we control the number of channels our image contains going forward. In the first block, the image's number of channels is set to 64. After this, every block doubles the number of image channels. Controlling the number of image channels provides control over the image's information beyond its condensed representation. Every channel for the image offers a new perspective on the encoded features and preserves information for the upsampling operation.

Overall, downsampling provides an efficient encoding for images, allowing network operation on lower dimensions. Additionally, each max pooling operation ensures that only the most important features are propagated through the encoding path, especially useful for semantic segmentation. 

#### Bottleneck

The bottleneck, or bridge, of our architecture serves as a path between our encoder and decoder stages. It follows the same process as the descending path: repeated convolution and activation functions to emphasize our encoded features prior to their upsampling. 

### Upsampling

Decoding our encoded features involves a similar process to the contracting path, in reverse. We have encoded our image to its most important features and now are solely intent on expanding their representation to resemble our original image dimensions. To decode our image representation, we perform transpose convolutions to upsample our representation before passing the features through another block of convolution and activation operations. The parameters for these operations are the exact same as for the encoding blocks (3x3 convolution, ReLU, repeat). At every decoding block, we are continuing to sweep our magnifying glass over the image, highlighting every 

Skip connections. Own section?

Importance of upsampling: reversing compression of encoder stage

#### Final Layer

## Training

## Impact
- image segmentation

- image classification

- image synthesis

- image restoration

- image superresolution
