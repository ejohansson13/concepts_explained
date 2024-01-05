![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)

The original U-Net research paper was released in 2015 and demonstrated improved success with biomedical image segmentation. The architecture's efficiency and performance with a small dataset announced its efficacy as a tool for computer vision tasks. In this page, we'll discuss the U-Net, its architecture, training process, and various implementations. This page assumes familiarity with machine learning technology and operations. If you don't have any machine learning experience, check out my [other page on the U-Net](https://github.com/ejohansson13/concepts_explained/blob/main/UNet/UNet.md), which makes no assumptions of machine learning experience.

## Architecture

The U-Net is an encoder-decoder architecture, deriving its name from the signature U-shape of its encoder and decoder paths. The encoder path represents the descent of the U-shape and downsamples the image, condensing it to its most important features. This compact representation, preserving image features, is then upsampled and ascends the network, with the output image replicating the original image's dimensions. Skip connections stretch across the symmetry of the network, providing additional context for upsampling. This added context, connecting equivalent stages in the architecture across paths, boosts the image reconstruction accuracy.

### Downsampling

We can consider the encoder path as consisting of a sequence of blocks. This sequence continues until we arrive at the end of the encoding path at the bottleneck. The blocks have the same composition: a convolution with a 3x3 kernel of stride 1 and 0 padding, followed by an elementwise ReLU activation function, another convolution layer with the same parameters as the first, another ReLU function, followed by a 2x2 max pooling operation to downsample the image for the next block of operations. Our initial convolution in the block also dictates the number of channels used for the remainder of the block. For the first block, this takes our number of channels from the input image's default to 64. For every other block of operations, the number of channels is doubled after the first convolution operation. 

Overall purpose of downsampling: efficiency and emphasis of important features.
 Something about how increasing channels preserves contextual information.
 Maybe something about how convolution weights are decided through training?
 Mention of stride for max pooling?

#### Bottleneck

The bottleneck, or bridge, of our architecture serves as a path between our encoder and decoder stages. It follows the same process as the descending path: repeated convolution and activation functions

### Upsampling

Description of upsampling process.

Skip connections. Own section?

Importance of upsampling: reversing compression of encoder stage

#### Final Layer

## Training

## Implementations
