# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. The U-Net model outperformed previous [SOTA](https://github.com/ejohansson13/concepts_explained/blob/main/Acronyms.md) image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and, in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image processing applications.

![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path (Encoder)

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into smaller, more manageable areas of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. Following the original literature, we will be considering a 572x572x1 image as those are the dimensions considered in the above illustration taken from the research paper. These images were actually 512x512, but the last 30 pixels around the edges of each image were mirrored to fulfill a 572x572 image size. The authors did not provide an explicit reason for this, but it can be inferred that this was to mitigate the loss of border pixels in each convolution operation. Allowing for input images to become much smaller than 28x28 likely would have resulted in information loss. Following the illustration, the image is passed through two sequential operations involving a 3x3 convolution and the ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I will explain what each of these concepts mean.  

### Convolution
Convolution is the application of a filter to an input matrix for the purpose of highlighting relevant features. The input matrix will be the image, where each image pixel is represented by a cell in our matrix. Repeated applications of these convolutional filters allow us to distinguish the truly important features in an image. For the example below, the filter will only contain one kernel. Here, a 3x3 kernel will be applied to a 6x6 matrix representing the image. The entire convolutional operation can be seen directly below, but we'll soon examine each stage of the operation in more detail.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

It's important to mention that the kernel values will be learned through training of the network and will be updated throughout the learning process. We can initialize the kernel values, but the network itself will take responsibility to learn the best values for distinguishing important features in the training images. We take the 3x3 kernel given above and perform convolution between the kernel (highlighted here in yellow) and a matrix subset (highlighted in blue). All we are doing here is multiplying each element of our submatrix with the corresponding kernel element.

In the first stage, -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product between the kernel and matrix elements and the sum for this specific subset becomes the first element in our output matrix (highlighted in green). Here, that's 37. We repeat this throughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our convolutional kernel takes the role of the mirrors within the kaleidoscope that we rotate to better understand the glass we are looking at. 
<p align="center" width="100%">
  Initially, we see one stage of the picture. <br>
  <img src="/UNet/Images/cwc_first_stage.png" alt="First stage of a convolution operation between a matrix and a kernel" width="25%">
</p>

<p align="center" width="100%">
  We rotate a mirror and see the next stage. Our submatrix changes, shifting to the right by one element. <br>
  <img src="/UNet/Images/cwc_second_stage.png" alt="Second stage of a convolution operation between a matrix and a kernel" width="25%"> 
</p>
 
<p align="center" width="100%">
  We rotate the mirror and again our submatrix shifts by one element. <br>
  <img src="/UNet/Images/cwc_third_stage.png" alt="Third stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

<p align="center" width="100%">
  And again, completing the topmost row. For every step in our convolutional operation, the relevant matrix subset will be in blue.<br>
  <img src="/UNet/Images/cwc_fourth_stage.png" alt="Fourth stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

So far, we have only been looking at the top row of the kaleidoscope image. And so we shift the lens down slightly to the next stage. Accordingly, our submatrix will shift down one row, and we will repeat the above process for the next row in our matrix. A lot of the image will look the same but we have swapped the topmost row for the next row down. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We complete the second row, shift down, and perform the same operations on the third row in our matrix. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) We shift down another row and arrive at all the information our kaleidoscope has to offer and correspondingly all the information our kernel has taken from our input matrix. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output matrix is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we lose out on the edge most matrix elements. Convolution discards the edges of the image due to the incomplete context around those pixels, similar to our example.

#### Stride, Padding, and Kernel Size

There are options to remedy the loss of information around the border of images. We’ll touch on those options and other convolutional details in this section.

##### Stride
Stride determines how the kernel moves around our input matrix. In our example above, a 3x3 kernel filtered the 6x6 input matrix. The kernel shifted by one value as it maneuvered through the matrix. It operated with a stride of 1. Each submatrix that interacted with the kernel is highlighted in blue below.
<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_1.png" width="55%">
</p>

The kernel operated with a stride of 1, shifting by one column or row for each operation. If the kernel operated with a stride of 2, it would "skip" a column and operate on the next 3x3 submatrix. Let's look at the progression of a kernel with a stride of 3.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_3.png" width="55%">
</p>

The kernel starts with the same initial submatrix. Then, it shifts by 3 to the next submatrix. Reaching the end of the row, it shifts down. With a stride of 1, it would shift down by one row. With a stride of 3, it shifts down by three rows. That submatrix is convolved, before the kernel shifts horizontally by another 3 columns and arrives at the end of the input matrix. This leaves far fewer submatrices that interact with the kernel, affecting the output matrix size. We can visualize this below.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_stride_3_result.png" width="55%">
</p>

The initial convolution operation with a stride of 1 gave an output matrix of 4x4. With a stride of 3, the same convolutional kernel outputs a 2x2 matrix. Changing the stride changes the number of opportunities the kernel has to interact with the input matrix elements. With a stride of 3, it still touches every matrix element, but there are no overlapping values in the submatrices. Each 3x3 submatrix is selected and convolved, then a new submatrix is selected. 

In contrast, convolution with a stride of 1 had multiple overlapping values between submatrices. This allows the kernel to consider both the current window of data and its relation to the previous window. Values shared between submatrices offer the kernel a comprehensive view of both the current submatrix and its context of neighboring data. Lengthening the stride narrows the kernel's focus to a singular window at a time and minimizes the context gleaned from shared values between operations. 

##### Padding
Another convolutional element is padding. Padding also affects the size of the convolutional output. In our initial example, we convolve a 6x6 input matrix to a 4x4 output matrix. Some information on the border of the matrix is lost. The impact of values along the edge of the matrix are minimized as they have fewer options to interact with the kernel. To mitigate the loss, we can employ padding. Padding insulates the input matrix by appending it with rows and columns of additional data. The additional data increases the number of interactions between the border values and the convolutional kernel. We’ll look at two padding methods: mirroring and padding with zeros.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_mirror.png" width="35%">
</p>

Mirroring, as seen above, copies adjacent outer elements. The intuition behind mirroring is extending the matrix with identical values to those along the border, ensuring the padded values follow the same distribution as the original matrix values. In the example above, we padded by 1. We added 1 row on top of the matrix, 1 row along the bottom, 1 column to the left of our matrix, and 1 column to the right. We can pad by any number, up to duplicating the matrix height and width. Beyond that, there is no additional data to mirror. In the U-Net paper, input images were padded from 512x512 to 572x572 to preserve image feature dimensions as they were downsampled. The last 30 rows and columns of the input matrix were mirrored and padded.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_zeros.png" width="35%">
</p>

An alternative option is padding with zeros. Padding with zeros is demonstrated above. We append the input matrix with zeros along the border, extending the data to have a greater impact in convolutions. Padding with zeros diverges from mirroring in the emphasis placed along the border values. While mirroring emphasizes homogeneity in the extension of the input data, padding with zeros devalues the introduction of new values, directing the kernel's focus to the original matrix data.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_padding_results.png" width="50%">
</p>

The results of both padding operations when convolved with a 3x3 kernel are illustrated above.
Both output matrices have the same height and width as their input matrices. Padding by 1 preserves the dimensionality of the input and prevents any downsizing of data. Logically, the differences lie on the matrix edges. This is where the input matrices were padded and this is the location of their distinctions. The inner 4x4 matrices of both outputs are identical. They’re also equal to the original 4x4 output matrix from our convolution without padding, revisited below. 

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_original_result.png" width="40%">
</p>

Padding controls the height and width of the output matrix without affecting the core values propagated through the network. Controlling the amount padded to the input matrix controls the size of the output matrix. Padding offers a quick and easy solution to preserve dimensionality throughout convolutional operations. There is no concern of data distortion with padding. Padding symmetrically centers the input data and appends additional values to the edges of the matrix. Padding by too much propagates nonsensical values along the edges of the output matrices. For that reason, the amount of padding is normally proportional to the input matrix size. 

##### Kernel Size
The last convolutional variable we'll cover is kernel size. In the example above and the majority of the U-Net, convolution is done with 3x3 kernels. They offer a local context while limiting the number of values in each convolution. They also prevent overt downsizing of matrix dimensions. Let's look at the effect of increasing kernel size to 5x5.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_kernel_five_by_five.png" width="45%">
</p>

We perform convolution with a stride of 1 and no padding. Increasing the kernel size to 5x5 shrinks the output matrix from 4x4 to 2x2. Interacting with more values at each convolution requires fewer operations to interact with the entirety of the input matrix. It can also affect the progression of image features. We can visualize this by comparing the original output matrix from a 3x3 kernel to our new output from a 5x5 kernel.

<p align="center" width="100%">
  <img src="/UNet/Images/convolution_kernel_size_results.png" width="30%">
</p>

Comparing these matrices, the increased kernel size seems to have overemphasized some features and underemphasized others. Broadening the window for every convolution operation resulted in an unbalanced impression of some features in the data. Increasing the kernel size compacted the number of interactions between the input matrix and the convolutional kernel, changing the features advanced by the network.

Stride, padding, and kernel size can all change a convolutional output. As demonstrated above, altering any of these parameters can affect the features advanced by the network and their dimensionality. Unless explicitly mentioned, every convolution operation in the U-Net will involve a stride of 1, no padding, and a kernel size of 3x3.

### Rectified Linear Unit (ReLU)
Now that we thoroughly understand convolution, let's talk about activation functions. Continuing with our matrix example, we can take our output matrix and apply an element-wise activation function. An activation function takes in a value and acts like a security checkpoint at the airport. At the airport, if you have a bottle with liquid over a certain volume, you must empty it before continuing. Rules are in place and if you fall short of those rules, you alter your input before proceeding. Depending on the value input to the activation function, it may allow that value to pass unaffected or reject the value and replace it with 0. These actions will also change depending on the respective activation function. The rectified linear unit (ReLU) activation function allows all nonnegative values to pass, and rejects negative values, setting them to 0.
<p align="center" width="100%">
  <img src="/UNet/Images/relu_activation_function.png" alt="A graph demonstrating the Rectified Linear Unit activation function" width="25%">
</p>

Let's take the ouput matrix from our initial convolution example and see how it's affected by the ReLU activation function. After passing our output matrix through the ReLU activation function, we have the following matrix. As you can see, only negative values were affected.
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

By emphasizing the most relevant features in our image, we are also diminishing the less important features. The network becomes less concerned with discoloration or lighting of an image and focuses on the critical features of the image.

Following the convolution, ReLU, and now max pooling operations, the most relevant features of the image have been highlighted for the network to learn. Distilling our higher-dimension image to a lower-dimension representation allows for easier and faster computations, especially when our images aren't 4x4 as in the example above, but 568x568. With each max pooling operation, we decrease our total number of pixels by 75%, halving both the number of rows and the number of columns in our matrix. By halving our matrix both horizontally and vertically, we have arrived at a much more compact image representation. 

### Channels
Let's take a step back and revisit convolution. They have an important feature I didn't touch on, channels. Channels are the third dimension for our image matrices. Similar to how images have a height and width, they also have channels. Channels represent the number of distinct spaces offering information on our image. Think of channels as a stack of our images. Each version of the image in the stack is a channel. Each channel in our stack offers a different perspective on our image. 

One way to think of this is through the RGB color space. RGB images are stored with three channels: red, green, and blue. Each channel focuses on one color in the image. We can look at the below image of a lake separated to its respective red, green, and blue channels. One channel in our image focuses on the intensity of red in the image. Another focuses on the green in our image, while the third channel focuses on the blue.
<p align="center" width="100%">
  <img src="/UNet/Images/image_channels.png" alt="An example image broken down to its respective red, green, and blue channels." width="75%">
</p>

Since we know that each image is a matrix, we can also consider channels as a stack of matrices. Each matrix in our stack corresponds to one channel in our image. Similar to above, our image will have three channels, one for each of the RGB colors. Therefore, our stack will have three matrices. Each matrix has the same height, width, and number of cells. Each cell in our matrices corresponds to one pixel of our image. The value of each cell illustrates the magnitude of the channel-specific color in that pixel of our image. In the example below, these values will range from 0-1, with 0 demonstrating an absence of color and 1 representing the full magnitude of that color. As we can see, the upper-left pixel in our image appears to be fairly split between red and blue with a smaller emphasis on green. The bottom-left pixel appears to have a heavy red influence, but green and blue are also apparent in that image pixel.
<p align="center" width="100%">
  <img src="/UNet/Images/channels.png" alt="An image matrix with pixel values corresponding to its red, green, and blue channels." width="25%">
</p>

The examples above explain the concept of image channels by tying each channel to one of the RGB colors. However, channels don’t have to be restricted to the color space. Channels can represent any image feature, and often represent image information we take for granted visually, but are essential to a computer’s comprehension. Presenting an image in more channels offers more information on its features and gives the network more opportunities to learn image information.

The alternative to multiple channels for an image is only one channel. This is known as grayscale. If an image only has one channel, it lacks all of the other information we described. The channel is entirely devoted to the magnitude of gray in the image. A 0 in a pixel would represent white, and a 1 would represent black. Grayscale images only need one channel for information. When performing convolution, we control the number of channels in our output, allowing the network to broaden its image understanding. It can go beyond grayscale, and process multiple image features from different perspectives. In the paper, the first convolutional operation receives a grayscale image as input and converts it to 64 channels representing the image features. That diagram is presented below.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_first_conv.png" width="10%">
</p>

Every rectangle indicating the image features will have the height and width dimensions near the bottom of the rectangle and the number of channels above the rectangle. A 572x572x1 image is input and broadened to 570x570x64. Our input image only holds one channel, as the biomedical images the network was trained on are all in grayscale. If we were training on RGB images, we could feed in images with 3 channels (572x572x3) and still have a 570x570x64 sized output. Convolution allows total control of the number of channels in an output image. Let's take a look at how that works.

### Convolution with Multiple Channels

In our initial convolution example, we explained that our convolutional filter would only contain one kernel. This was a simplified example. For more complex examples, i.e. when dealing with images with multiple channels, a convolutional filter is a collection of kernels, with one kernel for each input channel. When changing the number of channels in an output image through convolution, one filter exists for each output channel. Let's consider a multi-kernel, multi-filter example, expanding our convolution example from earlier before scaling up to the dimensions used in the paper.

In our earlier convolution example, we treated a singular 6x6 matrix as a grayscale image. Now let's consider a two-channel image. Two 6x6 matrices will represent our image. Those matrices are given below, and will be highlighted in their respective colors throughout the illustration. Keep in mind this is an example, so the values for the image, convolutional kernels, and output are all arbitrary.
<p align="center" width="100%">
  <img src="/UNet/Images/two_channel_image.png" width="45%">
</p>

If we want to expand this image to 3 channels, we would have one filter for each output channel. Each filter would have one kernel for each channel of our input image. For us, that means each filter will have two kernels. That gives us three filters (one for each output channel), each with two kernels (one for each input channel). The filters are given below and will be highlighted in yellow throughout the example.
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

You'll notice we want a 4x4x3 output, but we currently have 6 channels. Each convolutional filter is responsible for one channel of our output image, so we sum across each filter. This is as simple as matrix addition and gives our expected image output of 4x4x3. That addition is illustrated below, along with the overall convolution result.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_total_conv_2.png" width="65%">
</p>
<p align="center" width="100%">
  <img src="/UNet/Images/unet_total_conv_3.png" width="70%">
</p>

We have transformed our 6x6x2 input matrix into a 4x4x3 output. Convolution allowed the broadening of our two-channel image into three channels, offering additional perspectives for the network to better understand our image. Let's consider a higher-dimension example, the first convolution operation in the paper, but treat our input as an RGB image. In the paper, this is an expansion of a grayscale 572x572x1 image to 570x570x64. Instead, we'll be treating it as an RGB image of size 572x572x3 convolved to 570x570x64.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_first_conv.png" width="10%">
</p>

This will be a very similar process to the one explained above. Again, we'll have one 3x3 kernel for each input channel. Since our input image is 572x572x3, we have 3 kernels per filter. We have one filter for each output channel of our convolved image. Our output is going to be 570x570x64, so we need 64 filters. This gives us 64 filters (one for each output channel), each with 3 (number of input channels) kernels of dimension 3x3. Exactly like the example given above, each kernel corresponds to one input channel and outputs one matrix. Each kernel's output is then summed with the outputs of other kernels in the same filter, giving us one output channel per filter.

Even though we've expanded our problem set from the initial example of one 6x6 matrix, the same rules apply. We can still control the stride, padding, or kernel size for any of these operations. However, as we've seen, changing these parameters can have a significant impact on the dimensions of our output matrix. These impacts would be compounded with the repeated convolutions applied to each image as it progresses through the network.

Convolution gives our network total control over the number of input and output channels. Each kernel corresponds to one input channel. Each filter corresponds to one output channel. Having a unique kernel for each image input channel allows the network to singularly determine the best parameters to highlight the image details contained within each channel. Having multiple kernels for each filter ensures that every output channel of our image contains an amalgamation of the information offered across every channel of our input image. This preservation of information throughout our convolutional operations plays a large role in the efficiency of the U-net and its success with small training sets.

Now that we understand convolution with multiple channels, we can better understand the importance of increasing channels while decreasing our data dimensions. Increasing the number of channels affords our network additional perspectives to digest image features. Compressing our images to smaller and smaller dimensions throughout the contracting path of the U-Net runs the risk of information loss. Doubling the number of channels after every downsampling operation mitigates that risk by augmenting the number of avenues available to the network to observe image features.

## Bridge
The stages described above (3x3 convolution, ReLU, 3x3 convolution, ReLU, 2x2 max pooling) are repeated multiple times before arriving at the bridge, the bottom of the U-shaped architecture. This is our link between the contractive path we have descended and the expansive path we will soon ascend. Our image is at its smallest dimensions. From our initial 572x572x1 matrix, we have arrived at a 32x32x512 representation. This is the output of the final max pooling operation (red arrow below) and serves as our input to the bridge.
<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architecture taken from the corresponding 2015 research paper" width="55%">
</p>

At these smaller dimensions, information preservation is critical. Our progress thus far, descending the contracting path and filtering the most important features, is redundant if information is lost at this bottleneck. Preserving relevant information from multiple perspectives was the motivation behind expanding the number of channels for our image features. We continue that process at the bridge, doubling our number of channels to 1024. Concurrently, we apply another convolution and activation function operation. This is the absolute bottom of our network. At this stage, we are focusing on the minutiae of our technique. You're practicing keeping your hands high running around the screen to catch the ball. You're staying on the tips of your toes in the act of catching the ball. You're training the flick of your wrist when releasing the ball for a shot. We are simultaneously practicing these micro details in 1024 different situations to determine the significant aspects of our technique we'll maintain when scaling our technique back up to the macro level. The U-Net is scrutinizing the image features that have been propagated to the bridge and retaining the features it considers essential. We apply one more convolution and activation function pairing before beginning the process of reassembling our image from its features and scaling back up to pixel-space. 

## Expansive Path (Decoder)
Throughout our encoder process, we performed multiple sequential operations. Convolutions were followed by an activation function, and multiple convolution-activation operations occurred before we downsampled our image features. The decoder section follows a similar process. We are now putting our techniques together in hopes of shooting the perfect shot, just like the network assembling the features it has learned from its training. Throughout the expansive path, we'll be scaling what we've learned. Rather than practicing catching the ball, setting our feet, and raising the ball to shoot individually, we will be practicing these skills together. The purpose of the encoder was to determine the most important image features and provide the network enough channels to inspect these features. The decoder's purpose is to amalgamate the information offered by each of these channels while restricting information loss. The decoder is responsible for rebuilding the image from the network's determined features and comparing the model output to our desired outcome. Learning at every stage of the decoder will be augmented through skip connections, which I'll cover below. 

After we arrived at the bottom of the U, our image features reached their smallest dimensions. Rather than continue downsampling, we begin upsampling and ascending the expansive path of the architecture. At some point, no matter how much you practice each technique individually, the only way to increase your proficiency with shooting coming off of a screen is to incorporate your improved individual techniques into the movement of shooting off of a screen. That is what we are doing here. We've distilled our task into its multiple separate techniques and now it is time to start putting it all together again and observing our improvement.

### Skip Connections
As we ascend the expansive path, we notice a significant change in the architecture from the contracting path. Skip connections, or connecting paths, offer an opportunity for our network to augment its learning at every decoding step through information from the corresponding encoding step. Skip connections link images at similar stages in their respective processes. These connections across the architecture boost our image understanding. Images from the contracting path are cropped and concatenated onto our expansive path images. Since images are taken from equivalent steps in their respective processes, they have an equal number of channels. Our expansive path images, immediately following upsampling (represented by the green arrow below), are augmented with their counterparts and the number of channels is doubled. Images from the contracting path are cropped so that they fit the size of their respective stage in the expansive path. In the illustration below, decoding stage images have dimensions of 392x392x64, represented as the blue half of the rectangle above the green arrow. Encoding stage images have dimensions of 568x568x64 and are cropped to match the height and width of their decoding stage counterparts. The crop is denoted by the dotted blue lines and the connecting path is illustrated by the gray arrow in the image below. After concatenating the two groups of image features together, we arrive at a 392x392x128 matrix representation. The concatenated contracting path image features are depicted as a white rectangle extending the expansive path image features.
<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architecture taken from the corresponding 2015 research paper" width="60%">
</p>

The benefit here is that by combining the features present at the encoder stage with those present at the decoder stage, we obtain a more complete understanding of the image. We augment the learned semantic features of our data at the decoding stage with the spatial data provided by their encoding stage counterparts. Image channels contribute to the network's image comprehension, and concatenating decoding stage channels with their encoding stage complements provides additional context on the proximity and proportionality of image features. By concatenating the encoder stage representations to our decoder stage, we gain information from a higher resolution image and allow for more accurate image reconstruction. 

Throughout our basketball analogy, we've been breaking down the act of shooting a basketball while running around a screen into smaller and smaller movements. Practicing these smaller techniques allowed us to focus wholly on their improvement. We reached the smallest movements at the bridge of the U-Net: keeping your hands high to catch the ball, the flick of your wrist, etc. Now, we're incorporating these techniques into the entire movement. There is a risk. Abruptly scaling your follow-through on a standing jump shot to a shot while decelerating, turning, and releasing runs the risk of information loss. Suddenly having to account for many more variables (slowing, turning, jumping) leads to less attention paid to the follow-through. We can mitigate this information loss by recounting the procession of events leading to a successful shot in-motion. We decelerate when we come to the screen. We begin turning our hips as soon as the ball hits our hands. We set our feet to jump. We rise, and release the ball. Remembering the broader context of these smaller techniques assuages their upscaling friction. It allows us to focus not just on the important movements we learned (releasing the ball correctly), but to integrate them seamlessly into the complete movement. Assimilating encoder-stage information mitigates the U-Net's information loss while upscaling. The decoder-stage information has been wholly attentive to the image features propagating through the network. The cropped encoder-stage features remind the network of the structural proximity of the image features. Consolidating the information present in both stages boosts the network's spatial awareness while maintaining its concentration on the most important image features. 

This concept is illustrated below, visualizing the learned semantic information present at the decoder stage, the spatial information present at the encoder stage, and the benefit of concatenating both stages. This illustration is taken from [a video](https://www.youtube.com/watch?v=NhdzGfB1q74) explaining the overall U-Net architecture and its functionality.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

### Up-Sampling
Upsampling is a two-stage approach: nearest neighbor interpolation followed by a 2x2 convolution. Nearest neighbor interpolation functions by expanding each image feature's footprint. We quadruple our matrix size by doubling the number of rows and doubling the number of columns in our data. We can convert a 2x2 matrix to a 4x4 matrix by doubling the representation of each value horizontally and vertically, as seen below.
<p align="center" width="100%">
  <img src="/UNet/Images/simple_upsampling.png" alt="Matrix example of simple upsampling operation" width="45%">
</p>

We quadruple every instance of our previous values to double our matrix's rows and columns, a quick and easy approach to increase our height and width dimensions. After descending the contractive path, and compacting the image information, ascending our expansive path is focused on restoring the image to its original dimensions, while maintaining the features discovered through the descent. Nearest neighbor interpolation offers a quick upsampling operation without affecting our learned features.
<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="30%">
</p>

Directly following our nearest neighbor operation, we perform 2x2 convolution. Let’s look at an example. In the diagram above, the number of channels remains the same between upsampling (green arrow) and concatenating the encoder stage images with the decoder stage images (gray arrow). Two steps are performed sequentially in the green arrow illustrated above. 

First, nearest neighbor interpolation is performed as described. Every matrix value is quadrupled, doubling the matrix dimensions and giving an upsampled representation of the image features. In the diagram above, that would double the 196x196x128 matrix to 392x392x128. The number of channels does not change. We're only changing the height and width dimensions of the image features.

Next, 2x2 convolution is performed to halve the number of channels. Convolution at this kernel size immediately filters the upsampled feature values. The features now have dimensions of 392x392x64, setting the stage for concatenation with the encoder-stage features arriving via skip connection. We concatenate the encoder stage matrices (white half of above rectangle) to the upsampled image features (blue half of above rectangle), arriving at the depicted 392x392x128 matrix. The concatenated image features then proceed to the next stage of convolution and activation functions.

### Convolution and ReLU
The U-Net, like any other neural network, is dependent on the extraction of features from our data. The extraction of local image features is accomplished through convolution. The intuition behind the success of convolution for computer vision tasks is pixel dependency. Coherent images have a lot of pixels that are similar to the pixels around them. After all, it'd be pretty difficult to understand an image if one pixel had little to no dependency to the pixels immediately surrounding it. Because pixels are dependent on each other to create an understandable visual, the local windows employed by convolution allow us to extract image features. We can analyze one area of an image, understand its relation to adjacent areas and, from the ground-up, create an understanding of our image. However, as we've seen in our exhaustive examples, convolution is just a linear combination of data. Activation functions complete our feature extraction team. As mentioned in our activation function section, you can combine as many linear relationships as you want in as complex a sequence as you'd like, you'd still arrive at a linear relationship. Activation functions are what allow for our nonlinear modeling. Using a higher-order relationship to model our features allows for more complex expression of the dependencies in the data.

Throughout our decoding path, the convolution and activation function operations repeat their purpose from the encoding path: extracting and emphasizing image features. The success of the U-Net in computer visions is largely predicated on its ["inductive bias for spatial information"](https://arxiv.org/pdf/2112.10752#page=3&zoom=100,66,96). That successful learning and spatial awareness is directly correlated to the reiteration of extracted features arriving via the skip connections. However, the reiterated information needs to be reintegrated into the network, which is performed through the repeated application of convolution and activation functions.

#### Final Layer (1x1 Convolution)
<p align="center" width="100%">
  <img src="/UNet/Images/unet_architecture.png" alt="A screenshot of the UNet architecture from its corresponding 2015 research paper" width="65%">
</p>

We've propagated our image through the network, arriving at our final location to output our segmented image and measure our success. We've upscaled our image to the correct height and width, roughly matching our original pixel-space image dimensions. However, we have an incorrect number of channels. We can resolve this through 1x1 convolution. Convolution with 1x1 kernels operates on a per-pixel level across every channel. It assimilates each channel’s pixel-specific information and outputs those values into the preferred number of output channels. In this case, receiving features with 64 channels and outputting 2 channels requires 2 convolutional filters. Each filter contains 64 kernels of size 1x1. [This video](https://www.youtube.com/watch?v=c1RBQzKsDCk) offers a great explanation on 1x1 convolutions, their utility, and use cases.

<p align="center" width="100%">
  <img src="/UNet/Images/unet_final_conv.png" alt="The final convolution operation taken from the Unet research paper" width="25%">
</p>

After feeding our image features through the network from start-to-finish, we are now ready to measure our performance. Was our network successful in picking up the relevant information of our image? Could the model correctly segment that information, highlighting the appropriate segmentation area? Was the information correctly upscaled? Does everything in our output image look proportional? Let's compare our output to the provided ground-truth image and quantify our success.

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, and we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somehow, somewhere, something went wrong. You weren't expecting to get it right on your first attempt, were you? Despite the time and energy spent practicing your technique, something was off. Maybe it was the positioning of your feet. Maybe it was your release point. This is a learning process. With time, you'll be able to adjust your shot as you learn what a good shot looks like versus a bad shot. That learning process is exactly what happens with neural networks.

<p align="center" width="100%">
  <img src="/UNet/Images/unet_output_diagram.png" width="70%">
</p>

After the model outputs its predicted segmentation image, we compare our model's image to the provided ground-truth image, as illustrated above. The ground-truth image is the correct, expected answer. Any difference between our model output and the ground-truth is considered the loss. The function comparing our model output is logically called the loss function. The U-Net's loss function is cross-entropy. To perform cross-entropy, we first need to perform the softmax function. We apply the softmax function across our channel's two images, funneling the result into cross-entropy to compute our overall loss.

<p align="center" width="100%">
  <img src="/UNet/Images/softmax_diagram.png" width="50%">
</p>

Softmax takes our network's output across two channels and converts the raw values to probabilities. It funnels the network's calculations into a likelihood comparing each channel's probability per pixel. These probabilities sum to 1, as you can see above. Softmax calculates the likelihood that channel 0 (no segmentation) or channel 1 (segmentation area) is dominant. This is pertinent, because our ground-truth image is also full of 1's and 0's. A 0 denotes no segmentation, while a 1 denotes an area of interest that should be segmented. 

<p align="center" width="100%">
  <img src="/UNet/Images/cross_entropy.png" width="50%">
</p>

Cross-entropy receives both the model output (converted to probabilities, thanks to the softmax function) and the ground-truth image (all integers, 0 or 1). It penalizes every pixel position for its distance from the correct label. If the softmax function predicts that channel 0 was dominant at a certain pixel with 0.9 probability, the model receives a slight loss. If the channel outputs a 0.3 probability, that loss is larger. With this approach, all image channels are encouraged to match the true image labels and incorrect predictions are punished. The loss value determines the magnitude of correction. If we shoot the ball and it hits the front rim, we can recognize our shot missed, but it was fairly close. If we fail to make contact with the rim, backboard, or any part of the hoop, we need to make a larger correction to our shot. The magnitude of our loss determines the weight of correction necessary to improve future predictions. The learning of the network, tied to that loss, is backpropagation.

Backpropagation is the feedback reception and adjustment a network undergoes in response to its performance. It is the key to any neural network’s success. Throughout the training process, the network spends its time practicing and learning its task. It predicts values then adjusts its predictions in response to the training data's true values. These adjustments take place in the convolutional kernels we've covered throughout this page. It is the network’s responsibility to determine kernel values. There are multiple kernels for each filter and multiple filters for multi-channel inputs. The network iteratively determines those values by backpropagating the loss between our predictions and the ground-truth. The larger the loss, the larger an adjustment made to those kernel values. The network does a lot of learning in its early stages. Your first few shots might not go anywhere near the hoop. With time, your shots grow in accuracy. With time, the network's predictions grow in accuracy, until we arrive at a successful model with outputs similar to the expected output. The network learns through backpropagating the success of its predictions compared to the ground-truth answers.

## Other

Below, we’ll touch on a pair of ideas adjacent to the U-Net and its training that contribute to its success with image data.

### Data Augmentation
<p align="center" width="100%">
  <img src="/UNet/Images/data_augmentation.png" alt="An example image showing data augmentation variations" width="50%">
</p>

When training on a limited set of images, as with biomedical image segmentation, it is important to maximize the value extracted from the training set. Data augmentation plays a large role in the success of the U-Net with biomedical image segmentation by increasing the number of potential data samples through geometric or elastic shifts. Data augmentation might flip the images horizontally or vertically, rotate, crop, or change the saturation of the images. The intent is to present the images in as many different conditions as possible, such that the network can identify images regardless of their surrounding environment. By presenting images in different environments, the network learns to identify objects regardless of their context.

### Dropout
Machine learning models quickly become familiar with training images. As a result, they can struggle with data that differs from the training set. This is known as overfitting. The network hones in on the patterns of its training data and fails to adapt to new data. To prevent overfitting, we can practice dropout regularization. The network is a collection of neurons and dropout randomly blocks neurons in the training process, allowing all neurons to contribute equally to the network's decision-making. The distribution of decision-making responsibility ensures all neurons contribute to the network output. This gives the network the best potential to adapt to new data presented to our model.

# The End

Thank you for reading! I hope you enjoyed this explanation of the U-Net, intended for readers without any background ML knowledge to understand the architecture and training process of the model. Feel free to check out some of my other model explanations in their respective folders!
