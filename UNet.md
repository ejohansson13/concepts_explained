# Introduction

The U-Net! Released in 2015, [the original literature covering the U-Net architecture](https://arxiv.org/abs/1505.04597) celebrated its success with biomedical image segmentation, a field lacking sufficient annotated data. With data augmentation, the U-Net model outperformed previous SOTA image segmentation networks despite having a training set of only 30 images. Researchers realized the efficiency of the U-Net could be implemented to combat other computer vision issues and in 2022, [High Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) was released and Stable Diffusion took the world by storm not long after. In this space, we will discuss the U-Net architecture, how it functions, and what makes it such a fantastic tool for image generation.
**insert unet image here**


# Architecture

As you can see in the above image, the U-Net derives its name from its signature U shape. Descending the U, we downsample our input image to a more condensed representation before upsampling it as we ascend the U-shaped structure. Think of it like learning to play basketball: before you can hope to catch-and-shoot running around a screen, you first have to learn the fundamentals. How do you set your feet for your shot? How do you keep the ball at head height to prevent it from getting blocked? How do you release the ball when shooting? Distilling a complex operation into a smaller, more manageable area of learning allows you to focus on the truly important features. Rather than trying to learn all of these techniques at once, at every step of the process we can learn a portion of the bigger picture and then put these steps together for a finished product at the end. We follow this practice with the U-Net. The input image enters the network. For this purpose, we will be considering a 572x572 image as those are the dimensions considered in the above illustration. The image is passed through two sequential 3x3 convolution layers followed by a ReLU activation function applied element-wise. For those of you not from an ML background, it might seem like I slipped into a different language. I'll explain what each of these concepts mean. A kernel of specified size (in this case 3x3) is applied to the 572x572 matrix of pixels representing our image. If you're a visual learner like me, I've included an image below.

**insert convolution image here**

We take the 3x3 kernel given above and perform convolution between the kernel and the portion of the matrix it is above at a given moment. The leftmost element of the first row of the kernel is multiplied by the leftmost element of the first row of the corresponding matrix. Then the center elements perform the same operation. Next, the rightmost elements. In this instance, 

-1(1) + 2(2) + -3(3) + 

4(0) + 5(1) + -6(0) +

7(6) + -8(5) + 9(4) = 37.

Then, we begin the process anew with the second rows of the kernels and matrix respectively, then the third rows. As you can see in the example our input matrix is 6x6 while our output matrix is 4x4. The reason for this decrease in size is that as we move the kernel around the input matrix, we inevitably lose out on the edges of the matrix.
