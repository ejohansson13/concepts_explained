# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. The U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image processing applications.

![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path (Encoder)

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into smaller, more manageable areas of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572x1 image as those are the dimensions considered in the above illustration. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I will explain what each of these concepts mean.  

### Convolution
Convolution is the application of a filter to an input matrix for the purpose of highlighting relevant features. Repeated applications of these filters, or kernels, allow us to distinguish the truly important features associated with an image. A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I have included an example below.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

It's important to mention that the kernel values will be learned through training of the network and will be updated throughout the learning process. We can initialize the kernel values, but the network will take responsibility itself to learn the best values for distinguishing important features in the training data. We take the 3x3 kernel given above and perform convolution between the kernel (highlighted here in yellow) and the subset of the matrix it is above at a given moment (highlighted in blue). All we are doing here is multiplying each element of our matrix subset with the corresponding kernel element. I have broken this down into multiple stages below.

In the first stage, it is -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product and the sum for this matrix subset becomes the first element in our output matrix (highlighted in green). Here, that is 37. We repeat this thoughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our convolutional kernel takes the role of the mirrors within the kaleidoscope that we rotate to better understand the glass we are looking at. 
<p align="center" width="100%">
  Initially, we see one stage of the picture. <br>
  <img src="/UNet/Images/cwc_first_stage.png" alt="First stage of a convolution operation between a matrix and a kernel" width="25%">
</p>

<p align="center" width="100%">
  We rotate a mirror and see the next stage. Our matrix subset changes. <br>
  <img src="/UNet/Images/cwc_second_stage.png" alt="Second stage of a convolution operation between a matrix and a kernel" width="25%"> 
</p>
 
<p align="center" width="100%">
  We rotate the mirror and our subset changes again. <br>
  <img src="/UNet/Images/cwc_third_stage.png" alt="Third stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

<p align="center" width="100%">
  And again, completing the topmost row. <br>
  <img src="/UNet/Images/cwc_fourth_stage.png" alt="Fourth stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

So far, we have only been looking at the top row of the kaleidoscope image. And so we shift the lens down slightly to the next stage. A lot of the image will look the same but we have lost the topmost row and gained another row instead. Here, we are performing the same elementwise multiplication between the kernel and our matrix subset and summing the products. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We complete the second row, shift down and perform the same operations. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) We shift down another row and arrive at all the information our kaleidoscope has to offer and correspondingly all the information our kernel has chosen to highlight from our input matrix. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we lose out on the edgemost matrix elements. This is intended for the U-Net architecture. The authors refer to it as the overlap-tile strategy, important for biomedical image segmentation as we only utilize pixels of the image where the full context is available in the input image. Here is an illustration from the research paper: 
<p align="center" width="100%">
  <img src="/UNet/Images/biomed_convolution_example.png" alt="Biomedical image segmentation example of convolution operation from U-Net research paper released in 2015" width="40%"
</p>

Convolution discards the edges of the image due to the incomplete context around those pixels, similar to our example.

### Rectified Linear Unit
Now that we have our output matrix, we apply an element-wise activation function. An activation function takes in a value and acts like a security checkpoint at the airport. At the airport, if you have a bottle with liquid over a certain volume, you must empty it before continuing. Rules are in place and if you fall short of those rules, you alter your input before proceeding. Depending on the value input to the activation function, it may allow it to pass unaffected or reject the value and replace it with 0. These actions will also change dependent on the respective activation function. The rectified linear unit (ReLU) activation function allows all nonnegative values to pass, and rejects negative values, setting them to 0.
<p align="center" width="100%">
  <img src="/UNet/Images/relu_activation_function.png" alt="A graph demonstrating the Rectified Linear Unit activation function" width="25%">
</p>

After passing our output matrix through the ReLU activation function, we have the following matrix. As you can see, only negative values were affected.
<p align="center" width="100%">
  <img src="/UNet/Images/matrix_after_activation.png" width="55%">
</p>

By passing our output matrix through this activation function, we are zeroing all negative values. This is important. Activation functions take on the nonlinear responsibility of our network. For those of you with an ML background, this is intuitive. For others, I'll give a brief overview and attach some resources for further reading. Without introducing any nonlinearity, we are bounding our network to linear representations. Regardless of our architecture or number of layers, a combination of linear operations will always result in a linear output and fail to capture a more complex relationship.
<p align="center" width="100%">
  <img src="/UNet/Images/linear_vs_nonlinear.png" alt="A simple example of linear operations failing to capture more complex data relationships"               width="30%"
</p>
  
Expressing this idea in 2-dimensions might seem reductive, but we can see that regardless of the number of operations in our linear relationship, we fail to adequately represent the quadratic curve. We can better capture it at a single instance, but linear operations will always fail to correctly model nonlinear relationships. Nonlinear activation functions allow us to express more complex relationships for the network to better model and understand the data. [Here is a video of Andrew Ng on nonlinear activation functions](https://www.youtube.com/watch?v=NkOv_k7r6no). [And a blog post covering some activation functions with code examples](https://machinelearningmastery.com/using-activation-functions-in-neural-networks/).

### Down-sampling (Max Pooling)
The stages mentioned above are repeated twice. Our initial image is passed through a convolution operation, then ReLU, and that result is passed through another round of convolution and activation functions. Next, we arrive at the downsampling step, illustrated in the below diagram with a red arrow.
<p align="center" width="100%">
  <img src="/UNet/Images/first_downsampling_step.png" alt="The first max pooling operation performed on the contracting path of the U-Net" width="10%"
</p>

To downsample our matrix output, we perform a 2x2 max pooling operation. Max pooling maintains the most essential features of our image while diminishing our total information for faster computations. Preserving the most important features regardless of our matrix size builds robustness in the network to scale and orientation changes in images. We can take our previous matrix as an example. At each 2x2 matrix subset, we highlight the most relevant value and pass it on to our output matrix (highlighted in green).
<p align="center" width="100%">
  <img src="/UNet/Images/max_pooling.png" alt="Example of a max pooling operation transforming a 4x4 matrix into a 2x2 matrix" width="35%"
</p>

By emphasizing the most relevant features in our image, we are also diminishing the less important features. The network becomes less concerned with discoloration or lighting of an image and focuses on the critical features of the objects contained within the image.

Following the convolution, ReLU, and now max pooling operation, the most relevant features of the image have been highlighted for the network to learn. It has also arrived at a much more compact representation of the image, spotlighting the efficiency of the U-Net architecture. Distlling our higher-dimension image to a lower-dimension representation allows for easier and faster computations, especially when our images aren't 4x4 as in the example above, but 568x568. With each max pooling operation, we decrease our total number of pixels by 75% as we half both the number of rows and the number of columns in our matrix. By halving our matrix both horizontally and vertically, we have arrived at a much more compact image representation. 

### Channels
Let's take a step back and revisit convolution. They have an important feature I didn't touch on, channels. Channels are the third dimension for our image matrices. Similar to how images have a height and width, they also have channels. Channels represent the number of distinct spaces where our image offers information. The easiest way to think of this is through the RGB color space. RGB images are stored with three channels: red, green, and blue. Each channel contains information on its associated intensity. We can look at the image of a lake separated to its respective red, green, and blue channels.
<p align="center" width="100%">
  <img src="/UNet/Images/image_channels.png" alt="An example image broken down to its respective red, green, and blue channels." width="85%"
</p>

Each image channel is made up of the associated per-pixel intensity values. These matrices have the exact same height, width, and number of pixels. Each channel is a matrix whose values contain information on the specific pixel's intensity. In the example below, these values range from 0-1 and represent the intensity of that color in the associated pixel. The first pixel in the image appears to be fairly distributed between red and blue with a slight green influence. The bottom-left pixel appears to have a heavy red influence, but both green and blue coloring are also apparent in the pixel.
<p align="center" width="100%">
  <img src="/UNet/Images/channels.png" alt="An image matrix with pixel values corresponding to its red, green, and blue channels." width="35%"
</p>

However, channels don't have to be restricted to the color space. They can represent information on saturation, lighting, and many other visual effects we take for granted when seeing an image, but are crucial to a computer's comprehension. If the image only had one channel, it would lack information on color or other effects. This is known as grayscale. Instead of RGB coloring, the channel would contain information on the intensity of gray shading. One extreme of the intensity spectrum would be white, and the other would be black. Thus, grayscale images only need one channel for information.

When performing convolution, we can control the number of channels in our output, allowing the network to broaden its understanding of an image. It can go beyond grayscale, and process the image in a number of different spaces. These distinct spaces allow the network to accomplish its image processing goal. By incorporating a variety of information contained in different perspectives (channels), the computer gains a more complete awareness of the image it is viewing.

Convolution can affect our channel dimension, similar to max pooling's effect on height and width dimensions. We can take a grayscale image and perform convolution to broaden it to 64 channels, deepening the network's image comprehension. This is the example in the paper.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_channels.png" width="10%"
</p>

A 572x572x1 image is input and broadened to 570x570x64. Our input image only holds one channel, as the biomedical images the network was trained on are all in grayscale, offering information on shades of black-and-white but having no channels for color representation. If we were training on RGB images, we could feed in images with 3 channels and still have a 570x570x64 sized output. Throughout the network, our first convolution operation also dictates our number of channels moving forward. In the first stage, our first convolution operation gives us 64 channels. In the next stage, following our max pooling, we perform our first convolution operation and increase the channels to 128. This continues, doubling our number of channels in the first convolution operation of each stage until we arrive at the bottom of our U-shape and the bridge in our architecture. 

## Bridge
We repeat the above stages (3x3 convolution, ReLU, 3x3 convolution, ReLU, 2x2 max pooling) before arriving at the bridge, the bottom of the U-shaped architecture. This is our link between the contractive path we have descended and the expansive path we will soon ascend. Our image is at its smallest dimension size. From our initial 572x572x1 matrix, we have arrived at a 32x32x512 representation. This is the output of the final max pooling operation and serves as our input to the bridge.
<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="55%"
</p>

We repeat the process from throughout our contractive path descension. 3x3 convolution doubling our channel number, elementwise ReLU activation function, another 3x3 convolution, and ReLU takes our image dimensions to 28x28x1024. Since we have arrived at the bottom of the U, rather than downsample again, we upsample and begin our ascent up the expansive path of the architecture. At some point, no matter how much you practice each technique individually, the only way to increase your proficiency with shooting coming off of a screen is to incorporate your improved techniques into shooting off a screen. That is what we are doing here. We've distilled our task into its multiple separate techniques and now it is time to start putting it all together again and see how we have improved. 

## Expansive Path (Decoder)
Throughout our encoder process, we performed multiple sequential operations. Convolutions were followed by an activation function, and multiple convolution-activation operations occurred before we downsampled our matrix. We will follow a similar process with our decoder section. We are now putting our techniques together in hopes of shooting the perfect shot, just like the network assembling the features it has learned from its training. Rather than practicing catching the ball, setting our feet, and raising the ball to shoot individually, we will be practicing these skills together. Rather than breaking down our image into separate channels, we'll be accumulating the information we learn from these channels. Instead of downsampling, we will be upsampling. The purpose of the encoder was to determine the most decisive features from our image and condense them to a more computationally-friendly representation while losing minimal information. The decoder's purpose is to rebuild the network's understanding of the image from the determined features and compare model output to our desired outcome. Learning at every stage of the decoder will be augmented through [skip connections](#skip-connections), which I'll cover more below. 

### Skip Connections
As we ascend the expansive path, we notice a significant change in the architecture from the contracting path. Skip connections or connecting paths offer an opportunity for our network to learn at once from its decoding step while also receiving information from the corresponding encoding step. Skip connections link images at similar stages in their respective processes across the architecture to augment our learning structure. Images from the contracting path are concatenated on to our expansive path images. Since images are taken from equivalent steps in their respective processes, they have an equal number of channels. We are doubling the number of same-dimension images. Upsampled images are augmented with their counterparts and the number of channels is doubled. Imagine stacking layers of a cake. You'd want each layer to be the same size, and would cut each tier so that they evenly stack atop each other. In the diagram below, images taken from the contracting path are cropped so that they fit the size of the respective stage in the expansive path. The crop is denoted by the dotted blue lines and the connecting path by the arrow. 
<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="60%"
</p>

The benefit here is that by combining the features present at the encoder stage with those present at the decoder stage, we obtain a more complete understanding of the image. Every layer of a cake contributes to its overall taste, and every channel of our image contributes to the network's overall understanding. Our current decoder stage image is being reassembled after having been distilled through the descending path. It might have the general idea of an object's location, but not know what the object is. The cropped contracting path stage might be exactly aware of all objects, but not yet have highlighted them as objects for segmentation. By combining the two representations, we can recognize both where the object is and what object lies in the segmented portion of the image, boosting our network's overall perception of the image. An example is given below, taken from [this video](https://www.youtube.com/watch?v=NhdzGfB1q74) which does a phenomenal job explaining the overall U-net architecture.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

### Convolution, ReLU, and Up-Sampling
Immediately after the skip connection has concatenated our images atop one another, doubling our number of channels, we repeat a similar process to our contracting path. The first convolution stage receives as input our 104x104x512 concatenation of decoder and encoder stage images, and halves the number of channels. It outputs a 102x102x256 image stack, assimilating the information gained from the skip connection's concatenation. We then run our output matrix through an elementwise ReLU, and repeat with no further change to image channels.

The output matrix from the activation function is fed into our upsampling operation. When upsampling, we also half our number of channels to make space for the channels we will add through the connecting paths. 

We then repeat our convolution operation, followed by our activation function before upsampling to double our image resolution. When upsampling, we also half our number of channels to make space for the channels we will add through the connecting paths. We half our channels through upsampling, concatenate our images together, half our channels again through convolution, then perform convolution with the same number of channels, and feed into the next stage of upsampling. Concurrently, we double our image resolution through upsampling, keep it the same through our activation functions, then feed into the next stage of upsampling. This continues until we reach the final layer and output of our network.

### Up-Sampling
Two main approaches exist to upsampling: nearest neighbor interpolation and transpose convolution. I'll cover nearest neighbor interpolation here and transpose convolution in a later section of the document, as it is a more complex operation and not used in the original implementation of the network. Nearest neighbor interpolation is intuitive. We quadruple our matrix size by doubling the number of rows and doubling the number of columns in our data. We can convert a 2x2 matrix to a 4x4 matrix by doubling the representation of each value horizontally and vertically.
<p align="center" width="100%">
  <img src="/UNet/Images/simple_upsampling.png" alt="Matrix example of simple upsampling operation" width="65%"
</p>

We quadruple every instance of our previous values to double the number of rows and columns for our matrix. There are no kernels, learned values, or nonlinearity. This was the method used in the research paper and offers a quick path towards upsampling our compressed image representations. After descending the contractive path, and minimizing our image size, ascending our expansive path is focused on restoring the image to its original dimensions, while maintaining the features discovered through our descent of the network. Nearest neighbor interpolation offers a cheap upsampling operation without affecting our learned features.
<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="30%"
</p>

Directly following our upsampling operation, we perform 2x2 convolution. This can be seen in the diagram above as the number of channels remains the same between upsampling and concatenating the encoder stage images with the decoder stage images. Two steps are performed sequentially here. First, the nearest neighbor interpolation upsampling as described above, immediately followed by convolution with a 2x2 filter to make space for the cropped concatenated images. We could have a 196x196x128 matrix for our image, upsample to 392x392x128, convolve to 392x392x64, then double our channels through concatenation and arrive at a 392x392x128 representation of our data. Those operations are performed sequentially in the green upsampling arrow illustrated above. Concatenations are covered in more detail in the next section.

### Final Layer (1x1 Convolution)
<p align="center" width="100%">
  <img src="/UNet/Images/unet_architecture.png" alt="A screenshot of the UNet architecture from its corresponding 2015 research paper" width="50%"
</p>

After having performed the many associated concatenation, convolution, activation function, downsampling, and upsampling operations, we arrive at the final stage of our architecture. A 1x1 convolution operation is performed here to map the multiple channels for each image pixel to the desired number of channels for our output image. As seen in the image of the architecture, this could involve taking our 64 channels and performing convolution to output an image with 2 channels. [Andrew Ng has a great video](https://www.youtube.com/watch?v=c1RBQzKsDCk) on 1x1 convolutions, their utility, and use cases.

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, and we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somewhere in the process, something went wrong. Despite all the time and energy you've put into practicing your technique, something is still a little bit off. It's okay though! Maybe it was the positioning of your feet, maybe it was your release point, maybe you hadn't practiced enough with a defender and that threw off your shot. Whatever the reason, it's okay. This is a learning process and with time, you'll be able to adjust your shot as you learn more and more about what a good shot looks like and take fewer and fewer bad shots. That's exactly what happens with neural networks!

Backpropagation is key to the success of any neural network. It spends its time practicing and learning its task, and adjusts its predicted value to the true value provided by the training data. This feedback reception and adjustment is called backpropagation. In this case, the U-Net predicts its segmentations and finds out how good of a job it did. If it did a great job, it might go back and only slightly adjust its follow-through. If it did a really bad job, it might go back and do a serious rewrite of setting its feet and bringing the ball up to head height again. The feedback of the network is decided by its loss function. For the U-Net, those loss functions are Softmax and Cross-Entropy. 

Softmax measures the predicted activation of every pixel in our image across our channels. The channel with the most activation is then compared to the ground-truth image through cross-entropy. We measure the confidence of our network's output to the true result and backpropagate the correctness through our network. If the network was close to the true result, the model will only slightly change its convolution values. If the prediction was far off from the correct result, the model may take more drastic efforts to update its weights for more accurate future predictions. This process is repeated until we have exhausted our set of training images. 

## Other

### Data Augmentation
<p align="center" width="100%">
  <img src="/UNet/Images/data_augmentation.png" alt="An example image showing data augmentation variations" width="50%"
</p>
  
When training on a limited set of images, as with biomedical image segmenation, it is important to maximize the value we extract from our training set. Data Augmenation is one possibility. Data Augmentation performs a variety of operations on our images to build robustness in our model against new presentations of the same objects. We might flip our images horizontally, vertically, rotate, crop, or change the saturation of our images. After all, a bike will always be a bike. By presenting our images in various situations, our network learns to identify the object regardless of its context.

### Batch Normalization
The original paper does not include batch normalization, but it has become very common in subsequent architectures. Given a wide potential range of values included in the output matrix, we may want to take some action to stabilize the network and prevent it from being too affected by outlier values. One option is batch normalization. Batch normalization reduces the covariance, or joint variability of two variables. It is used to minimize the influence singular values may have on other values in the matrix. [Its PyTorch implementation can be found here](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). After performing our convolution operations, we can pass our output matrix through a batch normalization layer to concentrate our values and bring them closer to a uniform distribution. Uniform distributions become a lot easier to optimize for, than distributions spanning a wide range of values with outliers, which is what we had before. For this reason, batch normalization layers can aid in "simplifying" our values and bringing them closer to a standard distribution before feeding them to our activation layer. 

### Dropout
Our network is a large connection of multiple neurons connected together in an architecture resembling a U. To prevent our architecture from becoming overly dependent on any specific neurons, we practice dropout. By dropping out certain neurons, we prevent our network from becoming overly dependent on the roles of any specific neuron. This evenly distributes our weights throughout the network and helps us generalize to new data not included in our training set. If we have one neuron specifically attuned at recognizing bicycles in our image and are shown new images with jet skis, our network might be overeager to predict the jet ski is actually a bicycle due to the overvalued neuron. By practicing dropout, we prevent our network from being overly influenced by one neuron's behavior and distribute decision-making responsibilites among the many neurons present in our network. Dropout was not popularized in 2015, when the U-Net research paper was released but has quickly become a fundamental operation in neural networks since that time. Many newer implementations of the U-Net architecture include dropout functionality. 

### Transpose Convolution
Transpose convolution offers an alternative to nearest neighbor interpolation. It offers a learnable kernel to increase our spatial resolution to the desired dimensions. A brilliant illustration [can be found here](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) or videos approaching it from different perspectives can be found [here](https://www.youtube.com/watch?v=fMwti6zFcYY) and [here](https://www.youtube.com/watch?v=xoAv6D05j7g). We are creating a learnable kernel which pads our smaller matrix with zeros and performs convolution for an upsampled representation. Transpose convolution is a more complex operation and slightly more expensive in terms of both time and speed as a result. 

Imagine you have the perfect recipe for chicken wings. Unfortunately it only applies to five chicken wings and is enough to feed yourself for dinner every night, but you're having 10 friends over and want to increase the recipe to accomodate everyone. You could multiply the recipe by 10 to have enough food for yourself and your guests. This would be nearest neighbor interpolation. Alternatively, you could practice multiple times, changing the ingredients and playing with the spice levels until you arrive at a recipe you enjoy for 10 people. This would require multiple stages of practicing, tasting the wings, and rewriting the recipe until you're happy with the final product. This would be transpose convolution and has the associated time cost as well.

### Disclaimer: Additional Details
Image set is actually 512x512 pixels, converted to 572x572 by mirroring 30 pixels on either end of image. This concept is known as padding where an image size is slightly increased with placeholder values for ease of computation. Additionally, when performing convolution or max pooling, we are using strides to determine how much we move each filter along our input matrix. With a stride of 1, as in convolution, we only shift the matrix by one pixel when done with our operations. Max pooling will often use a stride of 2 and subsequently move the filter to values that were unseen in the previous operation. Think about it, if we have a 2x2 max pooling operation and then shift the filter over by 2, the next 2x2 matrix subset will be values that were not viewed in the previous operation.
