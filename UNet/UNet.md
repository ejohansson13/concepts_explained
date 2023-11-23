# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. With data augmentation, the U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image generation.



![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path (Encoder)

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into smaller, more manageable areas of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572 image as those are the dimensions considered in the above illustration. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I'll explain what each of these concepts mean.  

### Convolution
Convolution is the application of a filter to an input matrix for the purpose of highlighting relevant features. Repeated applications of these filters, or kernels, allow us to distinguish the truly important features associated with an image. A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I've included an image below.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

It's important to mention that the kernel values will be learned through training of the network and will be updated throughout the learning process. We can initialize the kernel values, but the network will take responsibility itself to learn the best values that allow it to best learn important features in the training data. We take the 3x3 kernel given above and perform convolution between the kernel and the portion of the matrix it is above at a given moment. All we are doing here is multiplying one element of our matrix subset with the corresponding kernel element. Here, it's -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product and the sum for this matrix subset becomes the first element in our output matrix. Here, that's 37. We repeat this thoughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our kernel takes the role of the mirrors within the kaleidoscope that we rotate to better understand the glass we're looking at. 
<p align="center" width="100%">
  Initially, we see one stage of the picture. <br>
  <img src="/UNet/Images/cwc_first_stage.png" alt="First stage of a convolution operation between a matrix and a kernel" width="25%">
</p>

<p align="center" width="100%">
  We rotate a mirror and see the next stage. <br>
  <img src="/UNet/Images/cwc_second_stage.png" alt="Second stage of a convolution operation between a matrix and a kernel" width="25%"> 
</p>
 
<p align="center" width="100%">
  We rotate the mirror again. <br>
  <img src="/UNet/Images/cwc_third_stage.png" alt="Third stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

<p align="center" width="100%">
  And again. <br>
  <img src="/UNet/Images/cwc_fourth_stage.png" alt="Fourth stage of a convolution operation between a matrix and a kernel" width="25%">  
</p>

Now imagine we've only been looking at the topmost of the kaleidoscope image. And so we shift the lens down slightly to the next stage. A lot of the image will still look the same but we've lost the topmost row of the image and gained another row instead. Here, we're simply performing the same elementwise multiplication between the kernel and our matrix subset and summing the products. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We shift down another row and perform the same operations. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) And another, where we've arrived at all the information our kaleidoscope has to offer and correspondingly all the information our kernel has chosen to highlight from our input matrix. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we lose out on the edgemost matrix elements. This is intended for the U-Net architecture. The authors refer to it as the overlap-tile strategy, important for biomedical image segmentation as we only utilize pixels of the image where the full context is available in the input image. Here's an example: 
<p align="center" width="100%">
  <img src="/UNet/Images/biomed_convolution_example.png" alt="Biomedical image segmentation example of convolution operation from U-Net research paper released in 2015" width="40%"
</p>

With training, the network was able to extract the important features from the image. The convolution operation also discarded the edges of the image due to the incomplete context around those pixels, similar to our example.

### Rectified Linear Unit
Now that we have our output matrix, we apply an element-wise activation function. An activation function takes in a value and acts like a security checkpoint at the airport. At the airport, if you have a bottle with liquid over a certain amount, you must empty it before continuing. Rules are in place and if you fall short of those rules, you alter your input before proceeding. Depending on the value input to the activation function, it may allow it to pass unaffected, apply a sinusoidal function, or reject the value and replace it with 0. The rectified linear unit (ReLU) activation function allows all nonnegative values to pass, and rejects negative values, setting them to 0.
<p align="center" width="100%">
  <img src="/UNet/Images/relu_activation_function.png" alt="A graph demonstrating the Rectified Linear Unit activation function" width="25%">
</p>

After passing our output matrix through the ReLU activation function, we have the following output.
<p align="center" width="100%">
  <img src="/UNet/Images/matrix_after_activation.png" width="55%">
</p>

By passing our output matrix through this activation function, we are zeroing all negative values. This is important. Activation functions take on the nonlinear responsibility of our network. For those of you with an ML background, this is intuitive. For others, I'll give a brief overview and attach some resources for further reading. Without introducing any nonlinearity, we are bounding our network to linear representations. Regardless of our architecture or number of layers, a combination of linear operations will always result in a linear output and fail to capture a more complex relationship.
<p align="center" width="100%">
  <img src="/UNet/Images/linear_vs_nonlinear.png" alt="A simple example of linear operations failing to capture more complex data relationships"               width="30%"
</p>
  
Expressing this idea in 2-dimensions might seem reductive, but we can see that regardless of the complexity of our linear relationship, we fail to adequately represent the quadratic curve. Non-linear activation functions allow us to express more complex relationships for the network to better understand the data. [Here's a video of Andrew Ng on nonlinear activation functions](https://www.youtube.com/watch?v=NkOv_k7r6no). [And a blog post covering the functions with some code examples](https://machinelearningmastery.com/using-activation-functions-in-neural-networks/).

### Down-sampling (Max Pooling)
This process is repeated twice. Our initial image is passed through a convolution operation, then ReLU, and that result is passed through another round of convolution and activation functions. Next, we arrive at the downsampling step.
<p align="center" width="100%">
  <img src="/UNet/Images/first_downsampling_step.png" alt="The first max pooling operation performed on the contracting path of the U-Net" width="10%"
</p>

To downsample our matrix output, we perform a 2x2 max pooling operation. Max pooling maintains the most essential features of our images while diminishing our total information for faster computations. Preserving the most important features regardless of our matrix size builds robustness in the network to any scale and orientation changes in images. We can take our previous matrix as an example. At each 2x2 step, we highlight the most relevant value and pass it on to our output matrix. By highlighting the most relevant features in our image, we are also diminishing the less important features. The network becomes less concerned in discoloration or lighting of an image and focused on the objects contained within the image.
<p align="center" width="100%">
  <img src="/UNet/Images/max_pooling.png" alt="Example of a max pooling operation transforming a 4x4 matrix into a 2x2 matrix" width="35%"
</p>

Following the convolution, ReLU, and now max pooling operation, the most relevant features of the image have been highlighted for the network to learn. It has also arrived at a much more compact representation of the image, highlighting the efficiency of the U-Net architecture. Distlling our higher-dimension image to a lower-dimension representation allows for easier and faster computations, especially when our images aren't 6x6 as in the example, but 572x572. With each max pooling operation, we decrease our total number of pixels by 75% as we half the number of rows **and** columns in our matrix. 

### Bridge
We would repeat the above stages thrice more (3x3 convolution, ReLU, 3x3 convolution, ReLU, 2x2 max pooling) before arriving at the bridge, the bottom of the U-shaped architecture. This is our link between the contractive path we've descended and the expansive path we will soon ascend. Our image is at its smallest dimension size. From our initial 572x572 matrix, we have arrived at a 32x32 representation. Here, we receive the output of the final max pooling operation as our input.
<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="55%"
</p>

We repeat the process from throughout our contractive path descension. 3x3 convolution, followed by the elementwise ReLU activation function is performed twice, taking our image size down to 28x28. Since we've arrived at the bottom of the U, rather than downsample again, we upsample and begin our ascent up the expansive path of the architecture. At some point, no matter how much you practice shooting from a stationary position, the only way to increase your proficiency with shooting coming off of a screen is to incorporate your improved technique into shooting off a screen. That's what we're doing here. We've distilled our task into its multiple separate techniques and now it's time to start putting it all together again and see how we've improved. 

## Expansive Path (Decoder)
Throughout our encoder process, we performed multiple sequential operations. Convolutions were followed by an activation function, and multiple convolution-to-activation-functions occurred before we downsampled our matrix. We will follow the same process with our decoder section with some notable differences. We're now putting our techniques together in hopes of getting the perfect shot coming around a screen. Rather than practicing catching the ball, setting our feet, and raising the ball to shoot individually, we'll be practicing these skills together. At each stage instead of breaking techniques down to their smallest representations, we'll be adding these representations together. Rather than downsampling, we'll be upsampling. Additionally, we'll be augmenting our learning with [skip connections](#skip-connections). I'll cover these topics more below. 

### Up-Sampling
Two main approaches exist to upsampling: nearest neighbor interpolation or transpose convolution. Nearest neighbor interpolation is intuitive. We convert a 2x2 matrix to a 4x4 matrix by doubling the representation of each value.
<p align="center" width="100%">
  <img src="/UNet/Images/simple_upsampling.png" alt="Matrix example of simple upsampling operation" width="55%"
</p>

We duplicate every instance of our previous value to double the number of rows and columns for our matrix. There are no learned values here, it is simple and easily done. This was the method used in the original research paper and offers a quick path towards upsampling our compressed image representations. 

Transpose convolution offers an alternative. It offers a learnable kernel to increase our spatial resolution to the desired dimensions. A brilliant illustration [can be found here](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) or videos approaching it from different perspectives can be found [here](https://www.youtube.com/watch?v=fMwti6zFcYY) and [here](https://www.youtube.com/watch?v=xoAv6D05j7g). We are creating a learnable kernel which pads our smaller matrix with zeros and performs convolution for an upsampled representation. Transpose convolution is a more complex operation and slightly more expensive in terms of both time and speed as a result. 

Imagine you have the perfect recipe for chicken wings. Unfortunately it only applies to five chicken wings and is enough to feed yourself for dinner every night, but you're having nine friends over and want to increase the recipe to accomodate everyone. You could multiply the recipe by 10 to have enough food for yourself and your guests. This would be nearest neighbor interpolation. Alternatively, you could practice multiple times, changing the ingredients and playing with the spice levels until you arrive at a recipe you enjoy for 10 people. This would require multiple stages of practicing, tasting the wings, and rewriting the recipe until you're happy with the final product. This would be transpose convolution and has the associated time cost as well.

### Skip Connections
As we ascend the expansive path, we notice a significant change in the architecture from the contracting path. Skip connections or connecting paths offer an opportunity for our network to learn at once from its stage in the ascending path while also receiving information from the corresponding stage in the descending path. The connecting paths link similar dimension images across the architecture to augment our learning process. Images from the contracting path are concatenated on to our expansive path stages. Imagine stacking cheese for a cheeseburger. You'd want each cheese slice to be the same size. We'd be cutting the sizes of different cheese slices so they can be evenly stacked atop each other. Images taken from the contracting path can be seen in the image to be cropped so that they fit the size of the same stage in the expansive path. 
<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="60%"
</p>

The benefit here is that by combining the features present at the encoder stage with those present at the decoder stage, we obtain a more complete understanding of the image. Our current decoder stage image representation has been so distilled that it might have the general idea of an object's location, but not know what the object is. By combining the two representations, we can gain an understanding of both where the object is and what object lies in the segmented portion of the image, boosting our network's overall understanding of the image. An example is given below, taken from [this video](https://www.youtube.com/watch?v=NhdzGfB1q74) which does a phenomenal job explaining the overall architecture for those with a background in ML.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

### Convolution, ReLU, and Up-Sampling
Immediately after the skip connection has concatenated our images atop one another, doubling our number of channels, we repeat the same process as in our contracting path. We pass the new image with multiple channels through a 3x3 convolution followed by an element-wise ReLU operation. This first convolution stage halves our number of channels, absorbing the information gained from the concatenating stage of the skip connection. We then repeat our convolution operation, followed by our activation function before upsampling to double our image resolution. Upsampling can be performed using either of the methods described above: nearest neighbor interpolation or transpose convolution. When upsampling, we also half our number of channels to make space for the channels we will add through the connecting paths. We half our channels through upsampling, concatenate our images together, half our channels again through convolution, then perform convolution with the same number of channels, and feed into the next stage of upsampling. Concurrently, we double our image resolution through upsampling, keep it the same through our activation functions, then feed into the next stage of upsampling. This continues until we reach the final layer and output of our network.

### Final Layer (1x1 Convolution)
<p align="center" width="100%">
  <img src="/UNet/Images/unet_architecture.png" alt="A screenshot of the UNet architecture from its corresponding 2015 research paper" width="50%"
</p>

After having performed the many associated concatenation, convolution, activation function, and upsampling operations, we arrive at the final stage of our architecture. A 1x1 convolution operation is performed here to map the multiple channels for each image pixel to the desired number of channels for our output image. As seen in the image of the architecture, this could involve taking our 64 channels and performing convolution to result in an image with 2 associated channels.

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somewhere in the process, something went wrong. Despite all the time and energy you've put into practicing your technique, something is still a little bit off. It's okay though! Maybe it was the positioning of your feet, maybe it was your release point, maybe you hadn't practiced enough with a defender and that threw off your shot. Whatever the reason, it's okay. This is a learning process and with time, you'll be able to adjust your shot as you learn more and more about what a good shot looks like and take fewer and fewer bad shots. That's exactly what happens with neural networks!

Backpropagation is key to the success of any neural network. It spends its time practicing and learning its task, and adjusts its predicted value to the true value provided by the training data. In this case, the U-Net predicts its segmentations and finds out how good of a job it did. If it did a great job, it might go back and only slightly adjust its follow-through. If it did a really bad job, it might go back and do a serious rewrite of setting its feet and practice bringing the ball up to head height again. How good of a job the network did is decided by its loss function. For the U-Net, that loss function is Cross-Entropy. 

Cross-Entropy functions 

## Other

### Data Augmentation
When training on a limited set of images, as with biomedical image segmenation, it is important to maximize the value we extract from our training set. Data Augmenation is one possibility. Data Augmentation performs a variety of operations on our images to build robustness in our model against new presentations of the same objects. We might flip our images horizontally, vertically, rotate, crop, or change the saturation of our images. After all, a bike will always be a bike. By presenting our images in various situations, our network learns to identify the object regardless of its context. 

### Batch Normalization
The original paper does not include batch normalization, but it has become very common in subsequent architectures. Given a wide potential range of values included in the output matrix, we may want to take some action to stabilize the network and prevent it from being too affected by outlier values. One option is batch normalization. Batch normalization reduces the covariance, or joint variability of two variables. It is used to minimize the influence singular values may have on other values in the matrix. [Its PyTorch implementation can be found here](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). After performing our convolution operations, we can pass our output matrix through a batch normalization layer to concentrate our values and bring them closer to a uniform distribution. Uniform distributions become a lot easier to optimize for, than distributions spanning a wide range of values with outliers, which is what we had before. For this reason, batch normalization layers can aid in "simplifying" our values and bringing them closer to a standard distribution before feeding them to our activation layer. 

### Dropout
Our network is a large connection of multiple neurons connected together in an architecture resembling a U. To prevent our architecture from becoming overly dependent on any specific neurons, we practice dropout. By dropping out certain neurons, we prevent our network from becoming overly dependent on the roles of any specific neuron. This evenly distributes our weights throughout the network and helps us generalize to new data not included in our training set. If we have one neuron specifically attuned at recognizing bicycles in our image and are shown new images with jet skis, our network might be overeager to predict the jet ski is actually a bicycle due to the overvalued neuron. By practicing dropout, we prevent our network from being overly influenced by one neuron's behavior and distribute decision-making responsibilites among the many neurons present in our network. Dropout was not popularized in 2015, when the U-Net research paper was released but has quickly become a fundamental operation in neural networks since that time. Many newer implementations of the U-Net architecture include dropout functionality. 

### Disclaimer: Additional Details
Image set is actually 512x512 pixels, converted to 572x572 by mirroring 30 pixels on either end of image. This concept is known as padding where an image size is slightly increased with placeholder values for ease of computation. Additionally, when performing convolution or max pooling, we are using strides to determine how much we move each filter along our input matrix. With a stride of 1, as in convolution, we only shift the matrix by one pixel when done with our operations. Max pooling will often use a stride of 2 and subsequently move the filter to values that were unseen in the previous operation. Think about it, if we have a 2x2 max pooling operation and then shift the filter over by 2, the next 2x2 matrix subset will be values that were not viewed in the previous operation.
