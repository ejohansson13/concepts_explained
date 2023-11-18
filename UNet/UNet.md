# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. With data augmentation, the U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image generation.



![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

## Contracting Path

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into a smaller, more manageable area of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572 image as those are the dimensions considered in the above illustration. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I'll explain what each of these concepts mean.  

### Convolution
A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I've included an image below.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

We take the 3x3 kernel given above and perform convolution between the kernel and the portion of the matrix it is above at a given moment. All we are doing here is multiplying one element of our matrix subset with the corresponding kernel element. Here, it's -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We add up each product and the sum for this matrix subset becomes the first element in our  output matrix. Here, that's 37. We repeat this thoughout our input matrix until we have a complete output. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our kernel are the mirrors within the kaleidoscope we can rotate to better understand the glass we're looking at. 
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

Now imagine we've only been looking at the topmost of the kaleidoscope image. And so we shift the lens down slightly to the next stage. A lot of the image will still look the same but we've lost the topmost row of the image and gained another row instead. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We shift down another row. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) And another, where we've arrived at all the information our kaleidoscope has to offer. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output matrix is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we inevitably lose out on the edgemost matrix elements. This is intended for the U-Net architecture. The authors refer to it as the overlap-tile strategy, important for biomedical image segmentation as we only utilize pixels of the image where the full context is available in the input image. 

### Rectified Linear Unit
Now that we have our output matrix, we apply an element-wise activation function. An activation function takes in a value and acts as a filter. Depending on the value received, the activation function may simply allow it to pass, apply a sinusoidal function, or reject the value entirely and replace it with 0. The rectified linear unit (ReLU) activation function allows all nonnegative values to pass, and sets all negative values to 0. 
![A graph demonstrating the Rectified Linear Unit activation function](/UNet/Images/relu_activation_function.png)
By passing our output matrix through this activation function, we are simply zeroing out all negative values. 

### Max Pooling

## Bridge

## Expansive Path
Words. 

### Up-Convolution

## Skip Connections
Words about concatenation b/t contracting path and expansive path images. Images from contracting path must be cropped because of pixels lost from convolution for expansive path. 

### Convolution and ReLU

### Final Layer (1x1 Convolution)

### Error Function (Cross-Entropy)
We've done it. We've practiced setting our feet coming around the screen, we've practiced our hand positioning, we've practiced our follow-through. We've spent time practicing each part of the technique separately and now it's time to put it all together. You run around the screen, catch the ball, shoot, and... CLANGGGG! Off front-rim. What happened? Somewhere in the process, something went wrong. Despite all the time and energy you've put into practicing your technique, something is still a little bit off. It's okay though! Maybe it was the positioning of your feet, maybe it was your release point, maybe you hadn't practiced enough with a defender and that threw off your shot. Whatever the reason, it's okay. This is a learning process and with time, you'll be able to adjust your shot as you learn more and more about what a good shot looks like and take fewer and fewer bad shots. That's exactly what happens with neural networks!

Backpropagation is key to the success of any neural network. It spends its time practicing and learning its task, and adjusts its predicted value to the true value provided by the training data. In this case, the U-Net predicts its segmentations and finds out how good of a job it did. If it did a great job, it might go back and only slightly adjust its follow-through. If it did a really bad job, it might go back and do a serious rewrite of setting its feet and practice bringing the ball up to head height again. How good of a job the network did is decided by its loss function. For the U-Net, that loss function is Cross-Entropy. 

Cross-Entropy functions 

## Other

### Data Augmentation

### Batch Normalization
The original paper does not include batch normalization, but it has become very common in subsequent architectures. Given a wide potential range of values included in the output matrix, we may want to take some action to stabilize the network and prevent it from being too affected by outlier values. One option is batch normalization. Batch normalization reduces the covariance, or joint variability of two variables. It is used to minimize the influence singular values may have on other values in the matrix. [Its PyTorch implementation can be found here](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). After performing our convolution operations, we can pass our output matrix through a batch normalization layer to concentrate our values and bring them closer to a uniform distribution. Uniform distributions become a lot easier to optimize for, than distributions spanning a wide range of values with outliers, which is what we had before. For this reason, batch normalization layers can aid in "simplifying" our values and bringing them closer to a standard distribution before feeding them to our activation layer. 

### Dropout
