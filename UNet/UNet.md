# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. The U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image processing applications.

![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path (Encoder)

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into smaller, more manageable areas of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572x1 image as those are the dimensions considered in the above illustration taken from the research paper. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I will explain what each of these concepts mean.  

### Convolution
Convolution is the application of a filter to an input matrix for the purpose of highlighting relevant features. Repeated applications of these filters, or kernels, allow us to distinguish the truly important features associated with an image. A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I have included an example below.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

It's important to mention that the kernel values will be learned through training of the network and will be updated throughout the learning process. We can initialize the kernel values, but the network will take responsibility itself to learn the best values for distinguishing important features in the training data. We take the 3x3 kernel given above and perform convolution between the kernel (highlighted here in yellow) and the subset of the matrix it is above at a given moment (highlighted in blue). All we are doing here is multiplying each element of our matrix subset with the corresponding kernel element. I have broken this down into multiple stages below.

In the first stage, -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product and the sum for this matrix subset becomes the first element in our output matrix (highlighted in green). Here, that's 37. We repeat this thoughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our convolutional kernel takes the role of the mirrors within the kaleidoscope that we rotate to better understand the glass we are looking at. 
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
  <img src="/UNet/Images/image_channels.png" alt="An example image broken down to its respective red, green, and blue channels." width="75%"
</p>

Each image channel is made up of the associated per-pixel intensity values. These matrices have the exact same height, width, and number of pixels. Each channel is a matrix whose values contain information on the specific pixel's intensity. In the example below, these values range from 0-1 and represent the intensity of that color in the associated pixel. The first pixel in the image appears to be fairly distributed between red and blue with a slight green influence. The bottom-left pixel appears to have a heavy red influence, but both green and blue coloring are also apparent in the pixel.
<p align="center" width="100%">
  <img src="/UNet/Images/channels.png" alt="An image matrix with pixel values corresponding to its red, green, and blue channels." width="25%"
</p>

However, channels don't have to be restricted to the color space. They can represent information on saturation, lighting, and many other visual effects we take for granted when seeing an image, but are crucial to a computer's comprehension. If the image only had one channel, it would lack information on color or other effects. This is known as grayscale. Instead of RGB coloring, the channel would contain information on the intensity of gray shading. One extreme of the intensity spectrum would be white, and the other would be black. Thus, grayscale images only need one channel for information.

When performing convolution, we can control the number of channels in our output, allowing the network to broaden its understanding of an image. It can go beyond grayscale, and process the image in a number of different spaces. These distinct spaces allow the network to accomplish its image processing goal. By incorporating a variety of information contained in different perspectives (channels), the computer gains a more complete awareness of the image it is viewing.

Convolution can affect our channel dimension, similar to max pooling's effect on height and width dimensions. We can take a grayscale image and perform convolution to broaden it to 64 channels, deepening the network's image comprehension. This is the example in the paper. Every rectangle indicating the image will have its height and width dimensions near the bottom of the rectangle and its number of channels above the rectangle.
<p align="center" width="100%">
  <img src="/UNet/Images/unet_channels.png" width="10%"
</p>

A 572x572x1 image is input and broadened to 570x570x64. Our input image only holds one channel, as the biomedical images the network was trained on are all in grayscale, offering information on shades of black-and-white but having no channels for color representation. If we were training on RGB images, we could feed in images with 3 channels (572x572x3) and still have a 570x570x64 sized output. Throughout the network, our first convolution operation also dictates our number of channels moving forward. In the first stage, our first convolution operation gives us 64 channels. In the next stage, following our max pooling, we perform our first convolution operation and increase the channels to 128. This continues, doubling our number of channels in the first convolution operation of each stage until we arrive at the bottom of our U-shape and the bridge in our architecture. 

## Bridge
We repeat the above process (3x3 convolution, ReLU, 3x3 convolution, ReLU, 2x2 max pooling) before arriving at the bridge, the bottom of the U-shaped architecture. This is our link between the contractive path we have descended and the expansive path we will soon ascend. Our image is at its smallest dimension size. From our initial 572x572x1 matrix, we have arrived at a 32x32x512 representation. This is the output of the final max pooling operation and serves as our input to the bridge.
<p align="center" width="100%">
  <img src="/UNet/Images/bridge.png" alt="Diagram of the bridge of the U-Net architure taken from the corresponding 2015 research paper" width="55%"
</p>

We repeat the process from throughout our contractive path descension. 3x3 convolution doubling our channel number, elementwise ReLU activation function, another 3x3 convolution, and ReLU takes our image dimensions to 28x28x1024. Since we have arrived at the bottom of the U, rather than downsample again, we upsample and begin our ascent up the expansive path of the architecture. At some point, no matter how much you practice each technique individually, the only way to increase your proficiency with shooting coming off of a screen is to incorporate your improved techniques into shooting off a screen. That is what we are doing here. We've distilled our task into its multiple separate techniques and now it is time to start putting it all together again and see how we have improved. 

## Expansive Path (Decoder)
Throughout our encoder process, we performed multiple sequential operations. Convolutions were followed by an activation function, and multiple convolution-activation operations occurred before we downsampled our matrix. We will follow a similar process with our decoder section. We are now putting our techniques together in hopes of shooting the perfect shot, just like the network assembling the features it has learned from its training. Rather than practicing catching the ball, setting our feet, and raising the ball to shoot individually, we will be practicing these skills together. Rather than breaking down our image into separate channels, we'll be accumulating the information we learn from these channels. Rather than downsampling, we will be upsampling. The purpose of the encoder was to determine the most decisive features from our image and condense them to a more computationally-friendly representation while minimizing information loss. The decoder's purpose is to rebuild the image from the network's determined features and compare model output to our desired outcome. Learning at every stage of the decoder will be augmented through skip connections, which I'll cover below. 

### Skip Connections
As we ascend the expansive path, we notice a significant change in the architecture from the contracting path. Skip connections, or connecting paths, offer an opportunity for our network to augment its decoding step learning through information from the corresponding encoding step. Skip connections link images at similar stages in their respective processes. These connections across the architecture boost our image understanding. Images from the contracting path are cropped and concatenated on to our expansive path images. Since images are taken from equivalent steps in their respective processes, they have an equal number of channels. Our expansive path images are augmented with their counterparts and the number of channels is doubled. In the illustration below, images from the contracting path are cropped so that they fit the size of the respective stage in the expansive path. The crop is denoted by the dotted blue lines and the connecting path by the arrow in the image below. The concatenated contracting path image is depicted as a white rectangle extending our expansive path image.
<p align="center" width="100%">
  <img src="/UNet/Images/connecting_path_crop.png" alt="Crop of the U-Net architure taken from the corresponding 2015 research paper" width="60%"
</p>

The benefit here is that by combining the features present at the encoder stage with those present at the decoder stage, we obtain a more complete understanding of the image. Every channel of our image contributes to the network's overall understanding and provides more context for the image we are reassembling. 

Our current decoder stage image is being reassembled after having been compacted through the encoder. It has been condensed to contain the most important features of our image, but may have lost some spatial awareness of object locations. When trying to reconstruct our image to a higher resolution, regaining the spatial information is crucial. By concatenating the encoder stage representations to our decoder stage, we gain information from a higher resolution image and allow for more accurate image reconstruction. It's like assembling Lego. The picture on the box is a much larger representation of the object you're trying to construct. When building your bricks, you're aware that the floodlights go on top of the fire station. But where is the top of the station? By consulting the image on the box, you can gain a better understanding of the proportion of your bricks and where exactly to place the construction's most important features. 

Similarly for the U-Net, we've identified the most important features, but when trying to reassemble our details to a higher resolution we might have lost their exact placement. Concatenation of the encoder-stage images helps us. The cropped encoder stage might have spatial awareness of all the objects in the image, but not yet understand each feature's importance. The decoder stage image will be aware of the features but have lost their exact location when upsampling from smaller image dimensions. The Lego box doesn't place any emphasis on the floodlights, but it tells you their exact position relative to the other bricks of your construction. We place emphasis on the floodlights when reconstructing the building, and consulting the cover of the box helps us determine their location when reassembling the bricks from scratch. By combining the encoder and decoder stage representations, we can recognize what objects are important and their exact location in relation to the other image pixels, boosting our network's overall image perception. A simplified example is given below, taken from [this video](https://www.youtube.com/watch?v=NhdzGfB1q74) which does a phenomenal job explaining the overall U-net architecture.

<img src="/UNet/Images/decoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/encoder_stage_sc.png" width="33%" /> <img src="/UNet/Images/combined_stage_sc.png" width="33%" />

### Up-Sampling
Two main approaches exist to upsampling: nearest neighbor interpolation and transpose convolution. Nearest neighbor interpolation is the original implementation covered in the research paper. Transpose convolution is another alternative, [summarized below](#transpose-convolution). Nearest neighbor interpolation is intuitive. We quadruple our matrix size by doubling the number of rows and doubling the number of columns in our data. We can convert a 2x2 matrix to a 4x4 matrix by doubling the representation of each value horizontally and vertically.
<p align="center" width="100%">
  <img src="/UNet/Images/simple_upsampling.png" alt="Matrix example of simple upsampling operation" width="45%"
</p>

We quadruple every instance of our previous values to double our matrix's rows and columns. There are no kernels, learned values, or nonlinearity, which offers a quick path to upsampling our compressed images. After descending the contractive path, and minimizing our image size, ascending our expansive path is focused on restoring the image to its original dimensions, while maintaining the features discovered through our descent. Nearest neighbor interpolation offers a cheap upsampling operation without affecting our learned features.
<p align="center" width="100%">
  <img src="/UNet/Images/upsampling_step.png" alt="The last upsampling operation performed on the expanding path of the U-Net" width="30%"
</p>

Directly following our nearest neighbor operation, we perform 2x2 convolution. In the diagram above, the number of channels remains the same between upsampling and concatenating the encoder stage images with the decoder stage images. Two steps are performed sequentially in the green arrow illustrated above. First, the nearest neighbor interpolation upsampling as described above, immediately followed by convolution with a 2x2 filter to halve the number of channels. This is necessary as the cropped images arriving via skip connection will double the number of channels again through concatenation. Using the example in the diagram, we could have a 196x196x128 matrix for our image, upsample to 392x392x128, then immediately convolve to 392x392x64. The image's number of channels is then doubled through concatenation and we arrive at a 392x392x128 representation of our image. These image dimensions then proceed to the next convolution operation.

### Convolution and ReLU
After upsampling and skip connections have concatenated our images on to one another, we pass them through a series of convolution and activation function operations. The first convolution stage receives as input our consolidated decoder and encoder stage images. It halves the number of channels, absorbing the information gained from the skip connections. This output matrix is passed through an elementwise ReLU, before we repeat another stage of convolution and activation function operations with no further effect to our number of channels.

The purpose of these blocks is similar to their purpose in the contracting path. The convolution emphasizes our important features and the activation function implements nonlinearity for modeling complexity. Let's reexamine our earlier convolution and activation function example. Even in this simplified example, the operations have a notable impact. Our initial matrix with no value greater than 6 has jumped to contain a much larger range of values, even with ReLU limiting any negatives.
<p align="center" width="100%">
  <img src="/UNet/Images/convolution_result_revisited.png" width="30%"
</p>

If we pass the matrix through another stage with the same convolutional kernel, we can observe a greater activation of the matrix values, with some jumping to triple digits. Even in this example, we see how convolution might emphasize certain features and devalue others. Our activation function ties the negative value to 0, indicating little important information for our network in this region. We've emphasized critical regions of our image and devalued regions with minimal information.
<p align="center" width="100%">
  <img src="/UNet/Images/convolution_next_step.png" width="50%"
</p>

What we're doing here is akin to sifting for gold. Gold panners will find lucrative riverbeds and pan through sediment to find their gold. The repeated agitation of sediment in the pan leads to gold settling at the bottom. With convolution, we know there's value in our image. The repeated application of our convolutional filters lets the dust and sediment separate itself from our gold: the important features that our network analyzes to make its decision. Convolution and the other network operations are our pans and brushes. The network determines the values of our convolution kernels, and their optimal implementation to interact with the other network operations. It works in concert with activation functions, skip connections, upsampling and downsampling operations to serve as the network's decision-makers on the important features in an image. Throughout training, these values are updated as the network realizes what produces the best results. It receives feedback on its performance and updates the values of its convolutional filters to improve future results. 

This example is only meant to reiterate how convolutional operations work. It's unlikely for any two filters to have the same values. Each filter's values are optimized by the network to highlight significant details of our image and devalue insignificant features. Additionally, the network operates on a much larger scale. Matrices are not 6x6, 4x4 or 2x2, they are anywhere from 28x28 to 572x572. This is why our encoder path condenses each image to a much smaller representation. It provides an efficient method to determine the most important features of our image, regardless of its dimensionality.

### Final Layer (1x1 Convolution)
<p align="center" width="100%">
  <img src="/UNet/Images/unet_architecture.png" alt="A screenshot of the UNet architecture from its corresponding 2015 research paper" width="55%"
</p>

After having performed the many associated concatenation, convolution, activation function, downsampling, and upsampling operations, we arrive at the final stage of our architecture. Now, our output needs to be understandable for the network to classify its performance. We map our image to the expected number of output channels with a 1x1 convolution. A 1x1 operation directly convolves our channels to the expected dimensions for performance evaluation. As seen in the architecture above, this could involve taking our 64-channel image and performing convolution to output an image with 2 channels. Notice that this convolution operation does not impact our height and width dimensions, it only affects our number of channels. The mechanics of this operation are explained below or [check out this great video](https://www.youtube.com/watch?v=c1RBQzKsDCk) on 1x1 convolutions, their utility, and use cases.
<p align="center" width="100%">
  <img src="/UNet/Images/1_1_convolution.png" alt="An example of 1x1 convolution casting a 64x64x192 matrix to 64x64x1" width="35%"
</p>

In this example, we have a 64x64x192 matrix convolved to a 64x64x1 output. This is the utility of convolutional operations. We can input an image with any number of channels and output a matrix with our preferred number of channels for evaluation. For every output channel, we have a specific 1x1x192 convolutional filter. The third dimension of this 1x1 kernel corresponds to the number of channels of our input matrix. If we had a 64x64x64 matrix, our convolutional kernel would be 1x1x64. 

Returning to our example, each 1x1 filter outputs a 64x64x1 matrix. This is how we control the number of channels in our output image. If we want an output image of 64x64x2, we would have 2 distinct 1x1x192 convolutional filters. If we want an output image of 64x64x3, we would have 3 distinct 1x1x192 convolutional filters. Similar to the other convolutional kernels throughout our network, the values of these kernels are learned through network training to produce the best results for our task. Now that our image has the expected dimensions, we can evaluate the performance of our network.

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, and we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somewhere in the process, something went wrong. Despite all the time and energy you've put into practicing your technique, something is still a little bit off. Maybe it was the positioning of your feet, maybe it was your release point, maybe you hadn't practiced enough with a defender and that threw off your shot. Whatever the reason, it's okay. This is a learning process and with time, you'll be able to adjust your shot as you learn more and more about what a good shot looks like and take fewer and fewer bad shots. That's exactly what happens with neural networks!

Backpropagation is key to the success of any neural network. It spends its time practicing and learning its task, and adjusts its predicted value to the true value provided by the training data. This feedback reception and adjustment is called backpropagation. In this case, the U-Net predicts its segmentations and finds out how good of a job it did. If it did a great job, it might go back and only slightly adjust its follow-through. If it did a really bad job, it might go back and do a serious rewrite of setting its feet and bringing the ball up to head height again. The feedback of the network is decided by its loss function. For the U-Net, those loss functions are Softmax and Cross-Entropy. 

Softmax measures the predicted activation of every pixel in our image across our channels. The channel with the most activation for a particular pixel is considered a 1. For all other channels, that pixel is considered 0. Cross entropy then compares every channel to the image's true labels and penalizes every pixel position with the incorrect label. With this approach, all image channels are encouraged to match the true image labels and incorrect labels are penalized. We compare the network's output to the true result and backpropagate the correctness through our network. If the network was close to the true result, the model will only slightly change its convolution values. If the prediction was far off from the correct result, the model may take more drastic efforts to update its weights for more accurate future predictions. This process is repeated until we have exhausted our set of training images. 

## Other

### Data Augmentation
<p align="center" width="100%">
  <img src="/UNet/Images/data_augmentation.png" alt="An example image showing data augmentation variations" width="50%"
</p>
  
When training on a limited set of images, as with biomedical image segmenation, it is important to maximize the value we extract from our training set. Data Augmenation is one possibility and plays a large role in the success of the U-Net with biomedical image segmentation. Data Augmentation performs a variety of operations on our images to build robustness in our model against new presentations of the same objects. We might flip our images horizontally, vertically, rotate, crop, or change the saturation of our images. The idea is to present the subject of the image in as many different conditions as possible, such that the network can identify our image subject regardless of the surrounding environment. After all, a bike will always be a bike. By presenting our images in various situations, our network learns to identify the object regardless of its context.

### Dropout
Machine learning models quickly become familiar with images included in the training set. As a result, they often struggle with data that differs from the training set. This is a common problem in machine learning, known as overfitting. The network comes to expect all future data to resemble the data it was trained on. To prevent our network from overfitting, we practice dropout. Our network is a collection of neurons and dropout randomly cancels neurons in the training process to allow all neurons to contribute equally to the network's decision-making. We don't want our network to become overly dependent on one neuron. Instead, we want the network to distribute its decision-making such that all neurons contribute to the network output. This gives us the best opportunity to adapt to new data presented to our model.

Think of our architecture as a human body. If you rigorously practice pushups, you are likely to successfully develop your pectoral, deltoid and tricep muscles. Your legs are likely going to be underdeveloped in comparison. When presented with a squat, you might struggle. By instead practicing exercises that work out more muscles in your body, you give yourself the best opportunity to succeed in any athletic endeavor. Dropout is similar. It randomly cancels neurons to ensure a full-body workout for our network. Rather than only practicing pushups, it occasionally cancels the working of your pectoral, deltoid, and tricep muscles. Instead, it might push your leg or back muscles to work. By preventing the overdevelopment of one muscle group, the network encourages a more balanced development. In turn, this balanced training builds strength in every neuron and leads to greater success when presented with new data.

### Transpose Convolution
Transpose convolution offers an alternative to nearest neighbor interpolation. It offers a learnable kernel to increase our spatial resolution to the desired dimensions. One explanation [can be found here](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) or videos approaching it from different perspectives can be found [here](https://www.youtube.com/watch?v=fMwti6zFcYY) and [here](https://www.youtube.com/watch?v=xoAv6D05j7g). We are creating a learnable kernel which pads our smaller matrix with zeros and performs convolution for an upsampled representation. Transpose convolution is a more complex operation and slightly more expensive in terms of both time and speed as a result. 

Imagine you have the perfect recipe for chicken wings. Unfortunately it only applies to five chicken wings and is enough to feed yourself for dinner every night, but you're having 10 friends over and want to increase the recipe to accomodate everyone. You could multiply the recipe by 10 to have enough food for you and your guests. This would be nearest neighbor interpolation. But, maybe extrapolating the recipe 10x causes a slight loss in the tanginess from the lime zest or in the sweetness from your honey. You could practice multiple times, changing the ingredients and playing with the spice levels until you arrive at a new recipe you enjoy for 10 people. This would require multiple stages of practicing, tasting the wings, and rewriting the recipe until you're happy with the final product. This would be transpose convolution and has the associated time cost in perfecting its recipe as well.

### Disclaimer: Padding in Convolution
Some details were abstracted through this explanation, including the size of our training set images. Our image set is actually 512x512 pixels, expanded to 572x572 by mirroring the last 30 pixels around the edge of the image. This method is known as padding where a matrix is extended to preserve the boundary information. Think about our approach to convolution. We lost the outer boundary of pixels for every convolution operation we performed. Only the pixels with surrounding context were passed through our convolutional filter. To ensure no edge information was lost in these calculations, we initially pad our 512x512 images to 572x572 by mirroring the 30 pixels around the edge of our image. Padding and stride are important details in convolution we didn't get a chance to explore while examining the U-Net. If you want to read more about them, I [suggest the following website](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html).

Thank you for reading! I hope you enjoyed this explanation of the U-Net, intended for readers without any background ML knowledge to understand the architecture and training process of the model. Feel free to check out some of my other model explanations in their respective folders!
