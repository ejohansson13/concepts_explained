# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. With data augmentation, the U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image generation.



![A screenshot of the UNet architecture from its corresponding 2015 research paper](/UNet/Images/unet_architecture.png)
# Architecture

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into a smaller, more manageable area of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572 image as those are the dimensions considered in the above illustration. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I'll explain what each of these concepts mean. 

## Convolution
Maybe put convolution in a separate README. 
A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I've included an image below.

![An end-to-end convolution example of convolution between a 6x6 matrix and a 3x3 convolutional layer](/UNet/Images/convolution_with_calculations.png)

We take the 3x3 kernel given above and perform convolution between the kernel and the portion of the matrix it is above at a given moment. All we are doing here is multiplying one element of our matrix subset with the corresponding kernel element. Here, it's -1(1) + 2(2) + -3(3) + 4(0) + 5(1) + -6(0) + 7(1) + -8(2) + 9(3) = 37. We then add up each product and the sum for this matrix subset becomes the first element in our  output matrix. Here, that's 37. Think of a kaleidoscope. We have an optical instrument which can be rotated to produce changing optical effects until we have the complete picture. Our input matrix is the colored glass at the bottom. Our kernel are the mirrors within the kaleidoscope we can rotate to better understand the glass we're looking at. Initially, we see one stage of the picture.<img src="/UNet/Images/cwc_first_stage.png" alt="First stage of a convolution operation between a matrix and a kernel" width="195">) 

We rotate a mirror and we see the next stage. <img src="/UNet/Images/cwc_second_stage.png" alt="Second stage of a convolution operation between a matrix and a kernel" width="195"> 

We rotate the mirror again. <img src="/UNet/Images/cwc_third_stage.png" alt="Third stage of a convolution operation between a matrix and a kernel" width="195">  

And again. <img src="/UNet/Images/cwc_fourth_stage.png" alt="Fourth stage of a convolution operation between a matrix and a kernel" width="195">  

Now imagine we've only been looking at the topmost of the kaleidoscope image. And so we shift the lens down slightly to the next stage. A lot of the image will still look the same but we've lost the topmost row of the image and gained another row instead. ![Second row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_second_row.png) We shift down another row. ![Third row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_third_row.png) And another, where we've arrived at all the information our kaleidoscope has to offer. ![Fourth row of a convolution operation between a matrix and a kernel](/UNet/Images/cwc_fourth_row.png) 
As you can see in the example, our input matrix is 6x6 while our output matrix is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we inevitably lose out on the edgemost matrix elements. This is intended for the U-Net architecture. The authors refer to it as the overlap-tile strategy, important for biomedical image segmentation as we only utilize pixels of the image where the full context is available in the input image. 
