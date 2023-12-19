Stable Diffusion was one of multiple text-to-image generation models to generate widespread public attention on the advancements of artificial intelligence and machine learning. Dall-E and Midjourney were initial 

## Architecure

Overarching encoder-decoder architecure. In between, you've got three main stages: diffusion model, denoising u-net, and conditioner (prompt: written, image, audio, multimodal capabilities). Encoder is responsible for encoding images to their more compact latent representations. Diffusion model is repsonsible for learning how to take images apart, adding Gaussian noise at sequential timesteps to understand how noise alters images and eventually how the process can be reversed. Scheduler is responsible for picking out a latent representation from the Gaussian noise cesspit that is created from our diffusion model. U-Net is responsible for taking noisy, latent representation from scheduler and denoising it (still maintaining latent space, maximizing efficiency) such that our image can quickly be generated with some further conditioning. The conditioning comes via our prompt. Considering the text-to-image synthesis example, we feed in an empty prompt for classifier-free guidance of our model before feeding in our wanted textual prompt. The empty and textual prompts are concatenated together and combined with our U-Net result via the cross-attention mechanism popularized from the Transformer architecture. This gives us our latent end-product. This latent space matrix is then passed through our decoder, upsampled, etc. etc. etc. to become a legitimate pixel-space image. Following that, you can pass it through super-resolution networks or whatever else you want to do.

## Training

We are pumping in as many images as possible. The theory is that with our diffusion model learning to dissassemble real images to white noise, it will also learn the reverse process and learn to generate new images from white noise. 

## Inference
