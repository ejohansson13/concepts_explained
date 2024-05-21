# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. The U-Net model outperformed previous [SOTA](https://github.com/ejohansson13/concepts_explained/blob/main/Acronyms.md) image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image processing applications.

![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path (Encoder)

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into smaller, more manageable areas of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572x1 image as those are the dimensions considered in the above illustration taken from the research paper. These images were actually 512x512, but the last 30 pixels around the edges of each images were mirrored to fulfill a 572x572 image size. The authors did not provide an explicit reason for this, but it can be inferred that this was to prevent the image from becoming too small at the end of the contracting path. Allowing for input images to become much smaller than 28x28 likely would have resulted in information loss. Following the illustration, the image is passed through two sequential operations involving a 3x3 convolution and the ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I will explain what each of these concepts mean.  

### Convolution
Convolution is the application of a filter to an input matrix for the purpose of highlighting relevant features. The input matrix will be our image, where each image pixel is represented by a value in our matrix. Repeated applications of these convolutional filters allow us to distinguish the truly important features in an image. For the example below, our filter will only contain one kernel but, depending on the convolutional input, a filter may have multiple kernels. Here, we'll be applying a 3x3 kernel to a 6x6 matrix representing our image. The entire convolutional operation can be seen directly below, but we'll examine each stage of the operation in more detail.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

It's important to mention that the kernel values will be learned through training of the network and will be updated throughout the learning process. We can initialize the kernel values, but the network will take responsibility itself to learn the best values for distinguishing important features in the training images. We take the 3x3 kernel given above and perform convolution between the kernel (highlighted here in yellow) and a matrix subset (highlighted in blue). All we are doing here is multiplying each element of our matrix subset with the corresponding kernel element.

In the first stage, -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product between the kernel and matrix elements and the sum for this specific subset becomes the first element in our output matrix (highlighted in green). Here, that's 37. We repeat this thoughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our convolutional kernel takes the role of the mirrors within the kaleidoscope that we rotate to better understand the glass we are looking at. 
<p align="center" width="100%">
  Initially, we see one stage of the picture. <br>
  <img src="/UNet/Images/cwc_first_stage.png" alt="First stage of a convolution operation between a matrix and a kernel" width="25%">
</p>

<p align="center" width="100%">
  We rotate a mirror and see the next stage. Our matrix subset changes, shifting to the right by one element. <br>
  <img src="/UNet/Images/cwc_second_stage.png" alt="Second stage of a convolution operation between a matrix and a kernel" width="25%"> 
</p>
 
<p align="center" width="100%">
  We rotate the mirror and our subset changes, again shifting by one element. <br>
  <img src="/UNet/Images/cwc_third_stage.png" alt="Third stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

<p align="center" width="100%">
  And again, completing the topmost row. For every step in our convolutional operation, the relevant matrix subset will be in blue.<br>
  <img src="/UNet/Images/cwc_fourth_stage.png" alt="Fourth stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

So far, we have only been looking at the top row of the kaleidoscope image. And so we shift the lens down slightly to the next stage. Accordingly, our matrix subset will shift down one row, and we will repeat the above process for the next row in our matrix. A lot of the image will look the same but we have swapped the topmost row for the next row down. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We complete the second row, shift down, and perform the same operations on the third row in our matrix. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) We shift down another row and arrive at all the information our kaleidoscope has to offer and correspondingly all the information our kernel has taken from our input matrix. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output matrix is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we lose out on the edgemost matrix elements. Convolution discards the edges of the image due to the incomplete context around those pixels, similar to our example.

#### Stride, Padding, and Kernel Size

Convolution is a more complex operation than the example presented above. Now that we've walked through a simplified example, let's touch on some more details of its functionality. 

##### Stride
Stride determines how our kernel moves around our input matrix. In our example above, we utilized a 3x3 kernel to filter our 6x6 input matrix. Our kernel shifted by one value as it maneuvred through our matrix. Our kernel operated with a stride of 1. The matrix subsets that interacted with our kernel are highlighted in blue below.
<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_1.png" width="55%">
</p>

We can see that our kernel interacted with one 3x3 submatrix before shifting and interacting with the adjacent 3x3 submatrix. This is a stride of 1. If our kernel operated with a stride of 2, it would "skip" a value and operate on the next 3x3 submatrix. Let's look at which submatrices would be used if our kernel operated with a stride of 3.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_3.png" width="55%">
</p>

Our kernel starts with the same initial submatrix. It then strides 3 values and selects the next submatrix. Reaching the end of the row, it shifts down. With a stride of 1, it would shift down by one row. But, our stride defines both how our kernel moves horizontally as well as vertically. With a stride of 3, we shift down by 3 values. Our next submatrix is selected. We then shift horizontally by another 3 values and arrive at the end of our input matrix. This leaves us with far fewer submatrices that interact with our kernel, which affects the size of our output matrix. We can visualize this below.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_3_result.png" width="55%">
</p>

Our initial convolution operation with a stride of 1 gave us an output matrix of 4x4. With a stride of 3, the same convolutional kernel outputs a 2x2 matrix. Changing our stride changes the number of times our kernel interacts with our input matrix elements. With a stride of 3, it still touches every matrix element, but there are no overlapping values in our submatrices. Each 3x3 submatrix is isolated, convolved, then dispatched for the next submatrix. In contrast, our convolution with a stride of 1 had multiple overlapping values between operations. This allows the kernel to consider both the current window of data and its relation to our previous window, offering a comprehensive view of both the current submatrix and its broader context of neighboring data. Lengthening the stride narrows the kernel's focus to a singular window at a time and eliminates the context gleaned from shared values between operations. 

##### Padding
Another convolutional element is padding. Padding also affects the size of our convolutional output. In our initial example, we convolve a 6x6 input matrix and output a 4x4 matrix. Some information on the border of our matrix is lost because of the lack of corresponding context. Values along the edge of our matrix are minimized as they have fewer options to interact with the kernel. In order to mitigate that information loss, we can employ padding. Padding insulates our input matrix by appending it with rows and columns of additional data. The additional data offers broader context for our border values and allows them to be accordingly absorbed by our convolution function. This data usually follows one of two functions: mirroring or zeroing.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_mirror.png" width="55%">
</p>

Mirroring, as seen above, involves copying the adjacent outer elements. The intuition behind mirroring is to extend the matrix with identical values to those along the border, ensuring the broader context extending our image follows the same distribution as our original matrix values. In our example above we padded by 1. We added 1 row on top of our matrix, 1 row along the bottom, 1 column to the left of our matrix, and 1 column to the right. We can pad by any number, up until duplicating the matrix height and width. Beyond that, there is no additional data to mirror. In the U-Net paper, input images to the network were 512x512 but were padded through mirroring to 572x572 to preserve coherency as the image features were downsampled. Increasing our image size by 15 in all directions requires mirroring the last 15 rows and columns of the input matrix.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_zeros.png" width="55%">
</p>

An alternative option for padding is to pad with zeros. As demonstrated above, padding with zeros is autological. We append our input matrix with zeros along the border, extending our data to allow for the border values to factor into our convolutional operation. Padding with zeros diverges from mirroring in the emphasis placed along the border values. While mirroring emphasizes homogeneity in the extension of our input data, padding with zeros prefers to attend to the matrix's original data. It extends our input matrix by zeros, increasing our height and width but not compromising our original data. Instead we append zeros to prevent the new data from conflicting with our original input.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_results.png" width="55%">
</p>

Above, we can see the results of either padding operation when convolved with our original 3x3 kernel. As you can see, in both cases, our output matrix has the same height and width as our input matrix. Padding by 1 preserves the dimensionality of our image and prevents the slight shrinkage of data that would otherwise occur. Additionally, you can see that the only disctinction between the output matrices lies along the edges. This is where we padded the image and logically this is where the difference is visualized. In fact, the inner 4x4 matrix are not only identical to each other, but identical to the original 4x4 output matrix we received from our convolution without padding, demonstrated below. 

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_original_result.png" width="55%">
</p>

Padding allows control over the height and width of our output matrix without affecting the core values propagated throughout our network. By controlling the amount we pad to our input matrix, we have direct control over the size of our output matrix. If we want to preserve our height and width, we can pad by 1. If we actually want to increase the size of our output matrix, we can simply pad by a larger number. Regardless of our padding, we conserve both the values at the center of our output matrix and the center of our image. We don't have to worry about padding distorting our data perspective, as padding symmetrically always centers our input data. However, padding by too much would propagate nonsensical values along the edges of our output matrix. For that reason, our amount of padding is normally relatively small in proportion to our overall input matrix size. 

##### Kernel Size
The last convolutional variable we'll cover in this section is kernel size. In both the example above and the majority of the U-Net, 3x3 kernels are used for convolution. 3x3 convolutions are fairly ubiquitous throughout machine learning architectures. They offer a local context without considering too many values for each operation. Convolution with a 3x3 kernel ensures that only adjacent values are considered at every step. It also prevents overt downsizing of our matrix dimensions. Let's look at an example with a 5x5 convolutional kernel.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_kernel_five_by_five.png" width="55%">
</p>

As we can see, increasing our kernel to 5x5 has resulted in a decrease in the size of our output matrix. Considering more values at a time results in fewer operations needed to consider the entirety of our input matrix. 

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_kernel_size_results.png" width="55%">
</p>

If we compare our output matrices between convolution with a 3x3 kernel and a 5x5 kernel, it's difficult to observe much similarity. Broadening the window for every convolution operation can result in an unbalanced impression of certain features in the data. Some features are overemphasized, while others seem to be underemphasized. 





Conclusion disclaimer: For the rest of this page, our convolution operations will be with a stride of 1 and padding 0, unless explicitly mentioned. 

### Rectified Linear Unit
Now that we thoroughly understand convolution, let's talk about activation functions. Continuing with our matrix example, we can take our output matrix and apply an element-wise activation function. An activation function takes in a value and acts like a security checkpoint at the airport. At the airport, if you have a bottle with liquid over a certain volume, you must empty it before continuing. Rules are in place and if you fall short of those rules, you alter your input before proceeding. Depending on the value input to the activation function, it may allow that value to pass unaffected or reject the value and replace it with 0. These actions will also change dependent on the respective activation function. The rectified linear unit (ReLU) activation function allows all nonnegative values to pass, and rejects negative values, setting them to 0.
<p align="center" width="100%">
  <img src="/UNet/Images/relu_activation_function.png" alt="A graph demonstrating the Rectified Linear Unit activation function" width="25%">
</p>

After passing our output matrix through the ReLU activation function, we have the following matrix. As you can see, only negative values were affected.
<p align="center" width="100%">
  <img src="/UNet/Images/matrix_after_activation.png" width="55%">
</p>

By passing our output matrix through this activation function, we are zeroing all negative values. This is important. Activation functions take on the nonlinear responsibility of our network. Without introducing any nonlinearity, we are bounding our network to linear representations. Regardless of our architecture or number of layers, a combination of linear operations will always result in a linear output and fail to capture a more complex relationship. This is illustrated in the graph below. We have a simple linear relationship (y=2x) and a more complex linear relationship (y=5(2(x-1)-2)-5). Both are attempting to model the quadratic relationship \(y= x^2\).
<p align="center" width="100%">
  <img src="/UNet/Images/linear_vs_nonlinear.png" alt="A simple example of linear operations failing to capture more complex data relationships"               width="30%">
</p>
  
Expressing this idea in 2-dimensions might seem reductive, but we can see that regardless of the number of operations in our linear relationship, we fail to adequately represent the quadratic curve. We can better capture it at a single instance, but linear operations will always fail to correctly model nonlinear relationships. Nonlinear activation functions allow us to represent more complex relationships in our data, a critical aspect of machine learning models. [Here is a video of Andrew Ng on nonlinear activation functions](https://www.youtube.com/watch?v=NkOv_k7r6no), explaining their functionality and importance if you want to learn more.

### Down-sampling (Max Pooling)
The stages mentioned above are repeated twice. Our initial image is passed through a convolution operation, then ReLU, and that result is passed through another round of convolution and activation functions. Next, we arrive at the downsampling step, illustrated in the below diagram with a red arrow.
<p align="center" width="100%">
  <img src="/UNet/Images/first_downsampling_step.png" alt="The first max pooling operation performed on the contracting path of the U-Net" 
        width="10%">
</p>

To downsample our matrix output, we perform a 2x2 max pooling operation. Max pooling maintains the most essential features of our image while condensing our information. Preservation of information while downsampling is crucial. Ultimately, our image will be condensed to a 28x28 representation. Any information lost during that compression will lead to poorer results at the final output of our model. Below, we can revisit our matrix example. To preserve size, let's keep the matrix after one convolution operation and activation function, rather than performing the dual operations used in the U-Net. At each 2x2 matrix subset, we will highlight the most relevant value and pass it on to our output matrix (highlighted in green).
<p align="center" width="100%">
  <img src="/UNet/Images/max_pooling.png" alt="Example of a max pooling operation transforming a 4x4 matrix into a 2x2 matrix" width="35%">
</p>

By emphasizing the most relevant features in our image, we are also diminishing the less important features. The network becomes less concerned with discoloration or lighting of an image and focuses on the critical features of the objects contained within the image.

Following the convolution, ReLU, and now max pooling operations, the most relevant features of the image have been highlighted for the network to learn. Distlling our higher-dimension image to a lower-dimension representation allows for easier and faster computations, especially when our images aren't 4x4 as in the example above, but 568x568. With each max pooling operation, we decrease our total number of pixels by 75%, halving both the number of rows and the number of columns in our matrix. By halving our matrix both horizontally and vertically, we have arrived at a much more compact image representation. 

### Channels
Let's take a step back and revisit convolution. They have an important feature I didn't touch on, channels. Channels are the third dimension for our image matrices. Similar to how images have a height and width, they also have channels. Channels represent the number of distinct spaces offering information on our image. Think of channels as a stack of our images. Each version of the image in the stack is a channel. Each channel in our stack offers a different perspective on our image. 

One way to think of this is through the RGB color space. RGB images are stored with three channels: red, green, and blue. Each channel focuses on one color in the image. We can look at the below image of a lake separated to its respective red, green, and blue channels. One channel in our image focuses on the intensity of red in the image. Another focuses on the green in our image, while the third channel focuses on the blue.
<p align="center" width="100%">
  <img src="/UNet/Images/image_channels.png" alt="An example image broken down to its respective red, green, and blue channels." width="75%">
</p>

Since we know that each image is a matrix, we can also consider channels as a stack of matrices. Each matrix in our stack corresponds to one channel in our image. Similar to above, our image will have three channels, one for each of the RGB colors. Therefore, our stack will have three matrices. Each matrix has the same height, width, and number of pixels. Each cell in our matrices corresponds to one pixel of our image. The value of each cell illustrates the magnitude of the channel-specific color in that pixel of our image. In the example below, these values will range from 0-1, with 0 demonstrating no magnitude and 1 representing absolute magnitude. As we can see, the upper-left pixel in our image appears to be fairly split between red and blue with a smaller emphasis on green. The bottom-left pixel appears to have a heavy red influence, but green and blue are also apparent in that image pixel.
<p align="center" width="100%">
  <img src="/UNet/Images/channels.png" alt="An image matrix with pixel values corresponding to its red, green, and blue channels." width="25%">
</p>

The examples above explain the concept of image channels by tying each channel to one of the RGB colors. However, channels don’t have to be restricted to the color space. Channels can represent any image feature, and often represent image information we take for granted visually, but are essential to a computer’s comprehension. Presenting an image in more channels offers more information on its features and gives the network more opportunities to learn image information.

The alternative to multiple channels for an image is only one channel. This is known as grayscale. If an image only has one channel, it lacks all of the other information we described. There is no information on color, saturation or anything besides the intensity of gray shading. A 0 in a pixel would represent white, and a 1 would represent black. Grayscale images only need one channel for information. When performing convolution, we control the number of channels in our output, allowing the network to broaden its image understanding. It can go beyond grayscale, and process multiple image features from different perspectives. In the paper, the first convolutional operation receives a grayscale image as input and converts it to 64 channels representing the image features. That diagram is presented below.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_first_conv.png" width="10%">
</p>

Every rectangle indicating the image features will have the height and width dimensions near the bottom of the rectangle and the number of channels above the rectangle. A 572x572x1 image is input and broadened to 570x570x64. Our input image only holds one channel, as the biomedical images the network was trained on are all in grayscale. If we were training on RGB images, we could feed in images with 3 channels (572x572x3) and still have a 570x570x64 sized output. Convolution allows total control of the number of channels in an output image. Let's take a look at how that works.

### Convolution with Multiple Channels

In our initial convolution example, we explained that our convolutional filter would only contain one kernel. This was a simplified example. For more complex examples, i.e. when dealing with images with multiple channels, a convolutional filter is a collection of kernels, with one kernel for each input channel. When changing the number of channels in an output image through convolution, one filter exists for each output channel. Let's consider a multi-kernel, multi-filter example, expanding our convolution example from earlier before scaling up to the dimensions used in the paper.

In our earlier convolution example, we treated a singular 6x6 matrix as a grayscale image. Now let's consider a two-channel image. We'll have two 6x6 matrices representing our image. Those matrices are given below, and will be highlighted in their respective colors throughout the illustration. Keep in mind this is an example, so the values for the image, convolutional kernels, and output are all arbitrary.
<p align="center" width="100%">
  <img src="/UNet/Images/two_channel_image.png" width="45%">
</p>

If we want to expand this image to 3 channels, we would have one filter for each output channel we hope to generate. We would need three filters. Each filter would have one kernel for each channel of our input image. For us, that means each filter will have two kernels. That gives us three filters (one for each output channel), each with two kernels (one for each input channel). The filters are given below and will be highlighted in yellow throughout the example.
<p align="center">
  <img src="/UNet/Images/unet_filter1.png" width="30%" />
</p>
<p align="center">
  <img src="/UNet/Images/unet_filter2.png" width="30%" />
</p>
<p align="center">
  <img src="/UNet/Images/unet_filter3.png" width="30%" />
</p>

Now, let's perform convolution with these three filters. Each kernel corresponds to one image input channel. The first kernel in each filter will only interact with the first image channel and the second kernel in each filter will only ever interact with the second image channel. Feeding in our image, we repeat the same convolutional process described above. To save space, I've abstracted the calculations, but feel free to work them out for yourself.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_conv_filter1.png" width="45%">
</p>

We move on to the second convolutional filter and repeat our convolution across both kernels. Each kernel interacts with one image channel and we output two matrices.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_conv_filter2.png" width="45%">
</p>

We repeat the process with our third and final filter, applying its two kernels across our input image.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_conv_filter3.png" width="45%">
</p>

We've taken our 6x6x2 image input and, through convolution, arrived at 6 4x4 matrices for our output. You can see these matrices below.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_total_conv_1.png" width="75%">
</p>

You'll notice we want a 4x4x3 output, but we currently have 6 matrices. Each convolutional filter is responsible for one channel of our output image, so we sum across each filter. This is as simple as matrix addition and gives our expected image output of 4x4x3. That addition is illustrated below, along with the overall convolution result.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_total_conv_2.png" width="65%">
</p>
<p align="center" width="100%">
  <img src="/UNet/Images/unet_total_conv_3.png" width="70%">
</p>

We have transformed our 6x6x2 input matrix into a 4x4x3 output. Convolution allowed the broadening of our two-channel image into three channels, offering additional perspectives for the network to better understand our image. Let's consider a higher-dimension example, the first convolution operation in the paper, but treat our input as an RGB image. In the paper, this is an expansion of a grayscale 572x572x1 to 570x570x64. We'll be treating it as an RGB image of size 572x572x3 convolved to 570x570x64.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_first_conv.png" width="10%">
</p>

This will be a very similar process to the one explained above. Again, we'll have one 3x3 kernel for each input channel. Since our input image is 572x572x3, we have 3 kernels per filter. We have one filter for each output channel of our convolved image. Our output is going to be 570x570x64, so we need 64 filters. This gives us 64 filters (one for each output channel), each with 3 (number of input channels) kernels of dimension 3x3. Exactly like the example given above, each kernel corresponds to one input channel and outputs one matrix. The output matrices are then summed within each filter, giving us the same number of output channels as number of filters.

Convolution gives our network total control over the number of input and output channels. Each kernel corresponds to one input channel. Each filter corresponds to one output channel. Having a unique kernel for each image input channel allows the network to singularly determine the best parameters to highlight the image details contained within each channel. Having multiple kernels for each filter ensures that every output channel of our image contains an amalgamation of the information offered across every channel of our input image. This preservation of information throughout our convolutional operations plays a large role in the efficiency of the U-net and its success with small training sets.

Now that we understand convolution with multiple channels, we can better understand the importance of increasing channels while decreasing our data dimensions. Increasing the number of channels affords our network additional perspectives to digest image features. Compressing our images to smaller and smaller dimensions throughout the contracting path of the U-Net runs the risk of information loss. Doubling the number of channels after every downsampling operation mitigates that risk by augmenting the number of avenues available to the network to observe image features.

## Bridge
The stages described above (3x3 convolution, ReLU, 3x3 convolution, ReLU, 2x2 max pooling) are repeated multiple times before arriving at the bridge, the bottom of the U-shaped architecture. This is our link between the contractive path we have descended and the expansive path we will soon ascend. Our image is at its smallest dimension size. From our initial 572x572x1 matrix, we have arrived at a 32x32x512 representation. This is the output of the final max pooling operation (red arrow below) and serves as our input to the bridge.
<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="55%">
</p>

At these smaller dimensions, information preservation is critical. Our progress thus far, descending the contracting path and filtering the most important features, is redundant if information is lost at this bottleneck. Preserving relevant information from multiple perspectives was the motivation behind expanding the number of channels for our image features. We continue that process at the bridge, doubling our number of channels to 1024. Concurrently, we apply another convolution and activation function operation. This is the absolute bottom of our network. At this stage, we are focusing on the minutiae of our technique. You're practicing keeping your hands high running around the screen to catch the ball. You're staying on the tips of your toes in the act of catching the ball. You're training the flick of your wrist when releasing the ball for a shot. We are simultaneously practicing these micro details in 1024 different situations to determine the significant aspects of our technique we'll maintain when scaling our technique back up to the macro level. The U-Net is scrutinizing the image features that have been propagated to the bridge and retaining the features it considers essential. We apply one more convolution and activation function pairing before beginning the process of reassembling our image from its features and scaling back up to pixel-space. 

## Expansive Path (Decoder)
Throughout our encoder process, we performed multiple sequential operations. Convolutions were followed by an activation function, and multiple convolution-activation operations occurred before we downsampled our image features. The decoder section follows a similar process. We are now putting our techniques together in hopes of shooting the perfect shot, just like the network assembling the features it has learned from its training. Throughout the expansive path, we'll be scaling what we've learned. Rather than practicing catching the ball, setting our feet, and raising the ball to shoot individually, we will be practicing these skills together. The purpose of the encoder was to determine the most important image features and provide the network enough channels to inspect these features. The decoder's purpose is to amalgamate the information offered by each of these channels while restricting information loss. The decoder is responsible for rebuilding the image from the network's determined features and comparing the model output to our desired outcome. Learning at every stage of the decoder will be augmented through skip connections, which I'll cover below. 

After we arrived at the bottom of the U, our image features reached their smallest dimensions. Rather than continue downsampling, we begin upsampling and ascending the expansive path of the architecture. At some point, no matter how much you practice each technique individually, the only way to increase your proficiency with shooting coming off of a screen is incorporating your improved individual techniques into the holistic movement of shooting off of a screen. That is what we are doing here. We've distilled our task into its multiple separate techniques and now it is time to start putting it all together again and observing our improvement.

### Skip Connections
As we ascend the expansive path, we notice a significant change in the architecture from the contracting path. Skip connections, or connecting paths, offer an opportunity for our network to augment its learning at every decoding step through information from the corresponding encoding step. Skip connections link images at similar stages in their respective processes. These connections across the architecture boost our image understanding. Images from the contracting path are cropped and concatenated on to our expansive path images. Since images are taken from equivalent steps in their respective processes, they have an equal number of channels. Our expansive path images, immediately following upsampling (represented by the green arrow below), are augmented with their counterparts and the number of channels is doubled. Images from the contracting path are cropped so that they fit the size of their respective stage in the expansive path. In the illustration below, decoding stage images have dimensions of 392x392x64. Encoding stage images have dimensions of 568x568x64 and are cropped to match the height and width of their decoding stage counterparts. The crop is denoted by the dotted blue lines and the connecting path is illustrated by the gray arrow in the image below. After concatenating the two groups of image features together, we arrive at a 392x392x128 matrix representation. The concatenated contracting path image features are depicted as a white rectangle extending the expansive path image features.
<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="60%">
</p>

The benefit here is that by combining the features present at the encoder stage with those present at the decoder stage, we obtain a more complete understanding of the image. We augment the learned semantic features of our data at the decoding stage with the spatial data provided by their encoding stage counterparts. Image channels contribute to the network's image comprehension, and concatenating decoding stage channels with their encoding stage complements provides additional context on the proximity and proportionality of image features. By concatenating the encoder stage representations to our decoder stage, we gain information from a higher resolution image and allow for more accurate image reconstruction. 

Throughout our basketball analogy, we've been breaking down the act of shooting a basketball while running around a screen into smaller and smaller movements. Practicing these smaller techniques allowed us to focus wholly on their improvement. We reached the smallest movements at the bridge of the U-Net: keeping your hands high to catch the ball, the flick of your wrist, etc. Now, we're incorporating these techniques into the entire movement. There is a risk. Abruptly scaling the concentration of your follow-through on a standing jump shot to a shot while decelerating, turning, and releasing runs the risk of information loss. Suddenly having to account for many more variables (slowing, turning, jumping) leads to less attention paid to the follow-through. We can mitigate this information loss by recounting the procession of events leading to a successful shot in-motion. We decelerate when we come to the screen. We begin turning our hips as soon as the ball hits our hands. We set our feet to jump. We rise, and release the ball. Remembering the broader context of these smaller techniques assuages their upscaling friction. It allows us to focus not just on the important movements we learned (releasing the ball correctly), but to integrate them seamlessly into the complete movement. Assimilating encoder-stage information mitigates the U-Net's information loss while upscaling. The decoder-stage information has been wholly attentive to the image features propagating through the network. The cropped encoder-stage features remind the network of the structural proximity of the image features. Consolidating the information present in both stages boosts the network's spatial awareness while maintaining its concentration on the most important image features. 

This concept is illustrated below, visualizing the learned semantic information present at the decoder stage, the spatial information present at the encoder stage, and the benefit of concatenating both stages' data together. This illustration is taken from [a video](https://www.youtube.com/watch?v=NhdzGfB1q74) explaining the overall U-Net architecture and its functionality.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

### Up-Sampling
Two main approaches exist to upsampling: nearest neighbor interpolation and transposed convolution. Nearest neighbor interpolation is the original implementation covered in the research paper and, like max pooling, is very intuitive. Transposed convolutions are an alternative approach, [summarized below](#transposed-convolution). Nearest neighbor interpolation functions by expanding each image feature's footprint. We quadruple our matrix size by doubling the number of rows and doubling the number of columns in our data. We can convert a 2x2 matrix to a 4x4 matrix by doubling the representation of each value horizontally and vertically, as seen below.
<p align="center" width="100%">
  <img src="/UNet/Images/simple_upsampling.png" alt="Matrix example of simple upsampling operation" width="45%">
</p>

We quadruple every instance of our previous values to double our matrix's rows and columns. There are no kernels, learned values, or nonlinearity, offering a quick path to upsampling our compressed image features. After descending the contractive path, and compacting our image information, ascending our expansive path is focused on restoring the image to its original dimensions, while maintaining the features discovered through our descent. Nearest neighbor interpolation offers a quick upsampling operation without affecting our learned features.
<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="30%">
</p>

Directly following our nearest neighbor operation, we perform 2x2 convolution. In the diagram above, the number of channels remains the same between upsampling (green arrow) and concatenating the encoder stage images with the decoder stage images (gray arrow). Two steps are performed sequentially in the green arrow illustrated above. 

First, nearest neighbor interpolation is performed as previously described. Every matrix value is quadrupled, doubling our matrix dimensions and giving us an upsampled representation of our image features. In the diagram above, that would double our 196x196x128 matrix to 392x392x128. Notice our number of channels has not changed. We're only affecting the height and width dimensions of our image features. Next, 2x2 convolution is performed to halve the number of channels. Convolution at this kernel size immediately filters our upsampled feature values. Continuing the example, our features would now have dimensions of 392x392x64. Convolution filters the upsampled values across the provided number of channels, setting the stage for concatenation with the encoder-stage features arriving via skip connection. We concatenate our encoder stage matrices (white half of rectangle above) to our upsampled image features (blue half of rectangle), arriving at the depicted 392x392x128 matrix. These image dimensions then proceed to the next stage of convolution and activation functions.

### Convolution and ReLU
Throughout the U-Net, our model's architecture has been dependent on the building blocks of convolution and ReLU functions. Convolution filters and emphasizes certain image features, while our activation functions introduce nonlinearity to model the image features. ... serve same function in decoder as in encoder ... image of blue arrows throughout architecture representing conv + activation functions ... max pooling and upsampling serve purpose of changing image dimensions, but conv+relu determine channels and feature importance ... summarize purpose of U-Net architecture ... conv+relu determine both semantic feature learning and spatial information learning, skip connections connect spatial information back from learned encoder stage to decoder stage ... cite SD paper referring to U-Net's "inductive bias for spatial information"

Repeat the funcitonality of conv+relu functions. Similarity to encoder stage blocks. 

Reorganize decoder section. Up-sampling -> convolution+ReLU -> skip connections.

#### Final Layer (1x1 Convolution)
<p align="center" width="100%">
  <img src="/UNet/Images/unet_architecture.png" alt="A screenshot of the UNet architecture from its corresponding 2015 research paper" width="65%">
</p>

We've propagated our image through the network, arriving at our final location to output our segmented image and measure our success. We've upscaled our image to the correct height and width, roughly matching our original pixel-space image dimensions. However, we have an incorrect number of channels. We can resolve this through convolution. Throughout our architecture, as mentioned in the previous section, sequential applications of convolution and ReLU functions were denoted by a blue arrow. The final operation in the diagram is illustrated with a disparate color. That arrow represents a singular convolution operation. Specifically, since we do not want to change the height and width of our image, a 1x1 convolution. Convolution employing 1x1 kernels allows for comprehensive control over the number of channels in our image output. In this case, receiving features with 64 channels and ouputting 2 channels solely requires 2 convolutional filters, each containing 64 kernels of size 1x1. [This video](https://www.youtube.com/watch?v=c1RBQzKsDCk) offers a great explanation on 1x1 convolutions, their utility, and use cases.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_final_conv.png" alt="The final convolution operation taken from the Unet research paper" width="25%">
</p>

Something about two-channel structure. 0 channel has 0s everywhere except segmentation area. 1 channel has 1s in segmentation area, 0s everywhere else.

After feeding our image features through the network from start-to-finish, we are now ready to measure our performance. Was our network successful in picking up on the relevant information of our image? Could the model correctly segment that information, highlighting the appropriate segmentation area? Was the information correctly upscaled? Does everything in our ouput image look proportional? Let's compare our output to the provided ground-truth image and determine how successful we were.

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, and we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somehow, somewhere in the process, something went wrong. You weren't expecting to get it right in your first attempt, were you? Despite the time and energy spent practicing your technique, something was off. Maybe it was the positioning of your feet or maybe it was your release point. This is a learning process. With time, you'll be able to adjust your shot as you learn more about what a good shot looks like and what a bad shot looks like. That learning process is exactly what happens with neural networks.

<p align="center" width="100%">
  <img src="/UNet/Images/unet_output_diagram.png" width="70%">
</p>

After the model outputs its predicted segmentation image, we compare our model's image to the provided ground-truth image, as illustrated above. The ground-truth image is the correct, expected answer. Any difference between our model output and the ground-truth is considered the loss. The function comparing our model output is logically called the loss function. The U-Net's loss function is cross-entropy. To perform cross-entropy, we first need to perform the softmax function. We apply the softmax function across our channel's two images... funneling the output into cross-entropy.

<p align="center" width="100%">
  <img src="/UNet/Images/softmax_diagram.png" width="50%">
</p>

Softmax takes our network's output across two channels and converts the raw values to probabilities. It funnels the network's calculations into a likelihood comparing each channel's probability per pixel. These probabilities sum to 1. With softmax, the network is calculating the likelihood that channel 0 (no segmentation) is dominant, or if channel 1 (segmentation area) is dominant. 

<p align="center" width="100%">
  <img src="/UNet/Images/cross_entropy.png" width="50%">
</p>

Cross-entropy compares those pixel probabilities to the ground-truth activations. In the example above, we can see that cross-entropy receives both the ground-truth image and model output. It returns a single value, describing the overall difference between the two.


Consult YT video on U-Net for another idea on explaining backpropagation. 3:25 mark.

Backpropagation is the feedback reception and adjustment a network undergoes in response to its performance. It is key to the success of any neural network. Throughout the training process, the network spends its time practicing and learning its task. It predicts values then adjusts its predictions in response to the training data's true values. In this case, the U-Net predicts its segmentations and finds out how good of a job it did. If it did a great job, it might go back and only slightly adjust its follow-through. If it did a really bad job, it might go back and do a serious rewrite of setting its feet and bringing the ball up to head height again. Backpropagation and its magnitude is decided by the network's loss function. For the U-Net, those loss functions are Softmax and Cross-Entropy. 

Cross entropy then compares every channel to the image's true labels and penalizes every pixel position with the incorrect label. With this approach, all image channels are encouraged to match the true image labels and incorrect labels are penalized. We compare the network's output to the true result and backpropagate the correctness through our network. If the network was close to the true result, the model will only slightly change its convolution values. If the prediction was far off from the correct result, the model may take more drastic efforts to update its weights for more accurate future predictions. This process is repeated until we have exhausted our set of training images. 

## Other

### Data Augmentation
<p align="center" width="100%">
  <img src="/UNet/Images/data_augmentation.png" alt="An example image showing data augmentation variations" width="50%">
</p>
  
When training on a limited set of images, as with biomedical image segmenation, it is important to maximize the value we extract from our training set. Data Augmenation is one possibility and plays a large role in the success of the U-Net with biomedical image segmentation. Data Augmentation performs a variety of operations on our images to build robustness in our model against new presentations of the same objects. We might flip our images horizontally, vertically, rotate, crop, or change the saturation of our images. The idea is to present the subject of the image in as many different conditions as possible, such that the network can identify our image subject regardless of the surrounding environment. After all, a bike will always be a bike. By presenting our images in various situations, our network learns to identify the object regardless of its context.

### Dropout
Machine learning models quickly become familiar with images included in the training set. As a result, they often struggle with data that differs from the training set. This is a common problem in machine learning, known as overfitting. The network comes to expect all future data to resemble the data it was trained on. To prevent our network from overfitting, we practice dropout. Our network is a collection of neurons and dropout randomly cancels neurons in the training process to allow all neurons to contribute equally to the network's decision-making. We don't want our network to become overly dependent on one neuron. Instead, we want the network to distribute its decision-making such that all neurons contribute to the network output. This gives us the best opportunity to adapt to new data presented to our model.

Think of our architecture as a human body. If you rigorously practice pushups, you are likely to successfully develop your pectoral, deltoid and tricep muscles. Your legs are likely going to be underdeveloped in comparison. When presented with a squat, you might struggle. By instead practicing exercises that work out more muscles in your body, you give yourself the best opportunity to succeed in any athletic endeavor. Dropout is similar. It randomly cancels neurons to ensure a full-body workout for our network. Rather than only practicing pushups, it occasionally cancels the working of your pectoral, deltoid, and tricep muscles. Instead, it might push your leg or back muscles to work. By preventing the overdevelopment of one muscle group, the network encourages a more balanced development. In turn, this balanced training builds strength in every neuron and leads to greater success when presented with new data.

### Transposed Convolution
Transpose convolution offers an alternative to nearest neighbor interpolation. It offers a learnable kernel to increase our spatial resolution to the desired dimensions. One explanation [can be found here](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) or videos approaching it from different perspectives can be found [here](https://www.youtube.com/watch?v=fMwti6zFcYY) and [here](https://www.youtube.com/watch?v=xoAv6D05j7g). We are creating a learnable kernel which pads our smaller matrix with zeros and performs convolution for an upsampled representation. Transpose convolution is a more complex operation and slightly more expensive in terms of both time and speed as a result. 

Imagine you have the perfect recipe for chicken wings. Unfortunately it only applies to five chicken wings and is enough to feed yourself for dinner every night, but you're having 10 friends over and want to increase the recipe to accomodate everyone. You could multiply the recipe by 10 to have enough food for you and your guests. This would be nearest neighbor interpolation. But, maybe extrapolating the recipe 10x causes a slight loss in the tanginess from the lime zest or in the sweetness from your honey. You could practice multiple times, changing the ingredients and playing with the spice levels until you arrive at a new recipe you enjoy for 10 people. This would require multiple stages of practicing, tasting the wings, and rewriting the recipe until you're happy with the final product. This would be transpose convolution and has the associated time cost in perfecting its recipe as well.

### Disclaimer: Padding in Convolution
Some details were abstracted through this explanation, including the size of our training set images. Our image set is actually 512x512 pixels, expanded to 572x572 by mirroring the last 30 pixels around the edge of the image. This method is known as padding where a matrix is extended to preserve the boundary information. Think about our approach to convolution. We lost the outer boundary of pixels for every convolution operation we performed. Only the pixels with surrounding context were passed through our convolutional filter. To ensure no edge information was lost in these calculations, we initially pad our 512x512 images to 572x572 by mirroring the 30 pixels around the edge of our image. Padding and stride are important details in convolution we didn't get a chance to explore while examining the U-Net. If you want to read more about them, I [suggest the following website](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html).

# Examples

Below are some examples of the U-Net's functionality from a self-trained U-Net on the following [dataset](https://molab.es/datasets-brain-metastasis-1/?type=metasrd). The dataset contains images of a metastasis in the brain from Patient 040102. More information can be found in the code subdirectory of the U-Net folder. The U-Net was provided high-resolution imaging of the patient's brain across multiple time points and slowly learned to segment the metastasis from the provided annotated segmentations before being evaluated on images it was not trained on.

<p align="center" width=100%>
  <img src="/UNet/Images/0172_img.png" width="15%" /> <img src="/UNet/Images/0172_msk.png" width="15%" /> <img src="/UNet/Images/0172_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0185_img.png" width="15%" /> <img src="/UNet/Images/0185_msk.png" width="15%" /> <img src="/UNet/Images/0185_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0205_img.png" width="15%" /> <img src="/UNet/Images/0205_msk.png" width="15%" /> <img src="/UNet/Images/0205_pred.png" width="15%" />
</p>

As you can see above, the model demonstrates some success in segmenting the larger instances in the brain, but lacks nuance. The provided ground-truth examples mirror a coastline, accounting for minute details in the metastasis area. The U-Net predicitions lack this detail, and favor a circular segmentation, likely resulting from the loss metrics the model was trained on and the minimal resources put towards training this model. Let's look at how the model performs with smaller segmentation areas. Does the struggle to capture detail in the segmentation area result in an inability to segment smaller instances?
<p align="center" width=100%>
  <img src="/UNet/Images/0457_img.png" width="15%" /> <img src="/UNet/Images/0457_msk.png" width="15%" /> <img src="/UNet/Images/0457_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0530_img.png" width="15%" /> <img src="/UNet/Images/0530_msk.png" width="15%" /> <img src="/UNet/Images/0530_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0531_img.png" width="15%" /> <img src="/UNet/Images/0531_msk.png" width="15%" /> <img src="/UNet/Images/0531_pred.png" width="15%" />
</p>

It's a mixed bag. Some smaller segmentation instances are captured well by the model, while it misses others entirely. The U-Net still favors a circular segmentation area, regardless of the size, for all predictions. The U-Net has demonstrated success with large and small segmentation areas. Its primary limitation seems to be its inability to capture the nuance of segmentation instances. Does the U-Net demonstrate any further issues in its segmentation predictions?

<p align="center" width=100%>
  <img src="/UNet/Images/0551_img.png" width="15%" /> <img src="/UNet/Images/0551_msk.png" width="15%" /> <img src="/UNet/Images/0551_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0552_img.png" width="15%" /> <img src="/UNet/Images/0552_msk.png" width="15%" /> <img src="/UNet/Images/0552_pred.png" width="15%" />
</p>

In both of the images above, the segmentation area is large. We might expect our model to provide an inadequate border of the area, similar to our previous examples. Instead, the model predicts a much smaller segmentation area. The model lacks confidence in predicting a larger segmentation area despite previously successful performances with similarly sized segmentation instances. Let's take a look through the lens of the U-Net and what the model receives as input to understand its decision-making. 

<p align="center" width=100%>
  <img src="/UNet/Images/z_0551_img.png" width="15%" /> <img src="/UNet/Images/0551_msk.png" width="15%" /> <img src="/UNet/Images/0551_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0552_img.png" width="15%" /> <img src="/UNet/Images/0552_msk.png" width="15%" /> <img src="/UNet/Images/0552_pred.png" width="15%" />
</p>

It's difficult to see any segmentation area and understandable why the model would struggle to correctly highlight the relevant area. Let's revisit our original three examples to view the correlation between model input and a more successful segmentation prediction.

<p align="center" width=100%>
  <img src="/UNet/Images/z_0172_img.png" width="15%" /> <img src="/UNet/Images/0172_msk.png" width="15%" /> <img src="/UNet/Images/0172_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0185_img.png" width="15%" /> <img src="/UNet/Images/0185_msk.png" width="15%" /> <img src="/UNet/Images/0185_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0205_img.png" width="15%" /> <img src="/UNet/Images/0205_msk.png" width="15%" /> <img src="/UNet/Images/0205_pred.png" width="15%" />
</p>

Medical imaging is a complex technology. Patient movement during examination and instrument calibration play important roles in the success of [medical scans](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5447676/). The examples above highlight the limitations of an automatic detection model dependent on low signal-to-noise ratio imaging inputs. This model was far from perfect, it was created solely to serve as an implementation example for the U-Net technology presented in this page. Developing a model for industry would require many more iterations, careful tuning, and, in such a high-risk domain, likely still be semi-automatic and reliant on human annotation for support. The model would certainly need to be improved, but so could the data input to the model.

## Improving Input Data

The industry shift to [data improvement over model improvement](https://www.enterpriseai.news/2021/10/08/ai-modeling-reinvented-its-time-to-shift-to-better-data-rather-than-just-building-better-models/) for machine learning represents a fundamental change in perspective. The early focus of machine learning models was to improve models' performance by tuning parameters, activation functions, and altering model architectures. Newer advancements in machine learning are increasingly focused on improving the model input for superior model output. [Dall-E 3](https://cdn.openai.com/papers/dall-e-3.pdf) focused on improved text captioning in the training data for greater prompt fidelity in the generated images. That paper theorized that previous text to image models' inability to holistically capture the sentiment of textual prompts arose from "noisy and inacurate image captions in the training dataset". They remedied the situation by training a custom image captioning model to improve textual captions of training images and highlighted the importance of data in achieving efficacious generative models. The lead author of that paper, James Betker, affirmed as much in his [personal blog](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/). Successful model architectures with non-detrimental hyperparameters trained for a long enough time converge to approximately equivalent representations of their dataset. At once this is exceedingly apparent and incredibly unintuitive. If the fundamental role of a model is to accurately represent its data, it makes sense that architecture, optimization, and various configurations become arbitrary. Successful models will successfully model their data. But, this requires a strong mentality shift from academic projects in machine learning where the same choices rendered arbitrary by generative model-level training are the cornerstones of successful training on smaller datasets. All of which is a lot of words to say that the successful training of smaller models, e.g. this U-Net trained on biomedical images, is predicated on configuration choices but, even more so on the data it is trained on.

Thank you for reading! I hope you enjoyed this explanation of the U-Net, intended for readers without any background ML knowledge to understand the architecture and training process of the model. Feel free to check out some of my other model explanations in their respective folders!
