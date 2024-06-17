# Schedulers

This page is intended to cover schedulers and the more popular scheduling algorithms utilized with Stable Diffusion models. These schedulers are  mathematical equations for the reverse diffusion process and adapted to neural network architectures for the purpose of image generation. This is intended to be a summary overview, and will focus on broad comparison between the research papers inspiring these algorithms. There are many more schedulers than those covered here. This page will abstract rigorous mathematical concepts and attempt to summarize each paper's contribution to the advancement of diffusion models.

All quotes, ideas, and images from the summarization of each research paper are taken from the accompanying paper unless cited otherwise.

## DDPM

The first of a generation of research papers re-examining diffusion models for image synthesis, [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) proposed that diffusion models were capable of synthesizing high-quality images and bridged [noise conditional score networks](https://arxiv.org/pdf/1907.05600) during training with annealed Langevin dynamics while sampling. Following the [2015 Sohl-Dickstein et. al paper](https://arxiv.org/pdf/1503.03585), diffusion models were Markov chains parameterized with Gaussian noise and trained to reverse the diffusion process to generate samples matching their training data. Sohl-Dickstein proved that, given small additions of Gaussian noise along the Markov chain, both the forward and reverse processes would express valid samples from the same underlying distribution.

The forward process of adding noise to an image was dependent on a predetermined, linear schedule of variances. Adhering to Markov chain principles, each state was solely dependent on the previous state. A benefit of expressing the forward process as a schedule of variances was its sampling ability of any arbitrary latent variable along the chain. Changing the notation of the forward variance and taking the cumulative product of the rewritten variance allowed succinct expression of any state of x dependent only on the initial state. This removed the sequential restrictions of a Markov chain in the forward process. This equation can be seen below. Mention alpha = 1-beta, alpha_bar = cumpord(alpha) and beta is forward process variance.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/nice_property.png" width="100%">
</p>

The negative log likelihood of the reverse process (removing noise to arrive at a clean signal) could be optimized to arrive at an objective function comparing Gaussians via KL divergence. This simplified objective function was then optimizable via MSE and easily compatible with neural network architectures. The rewritten likelihood can be seen below.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/ddpm_nll_optimized.png" alt="Optimized negative log likehood equation for DDPM" width="100%">
</p>

The forward process variances could either be learned through reparameterization or treated as hyperparameters. The authors elected to treat them as hyperparameters, fixing them to constant values, and allowing the  L<sub>T</sub> term to be treated as a constant. Determining the  L<sub>t-1</sub> term determined the relationship between the forward noising process and reverse generative process. Determining the reverse generative process then became dependent on learning its mean and variance. The authors opted to tie the reverse process variance to the forward process variance. This could be implemented by rewriting the reverse process variance as a standard deviation constant multiplied by the identity matrix, where the standard deviation at every timestep was the square root of the forward process variance. Include equation below. This was also optimal for x0 sampled from normal distribution with 0 mean and identity variance.

After determining that the L<sub>t-1</sub> term depended on the mean and variance of the denoising process, and accounting for the variance, the authors rewrote the L<sub>t-1</sub> term[Eq. 8]. The rewritten loss served to minimize the distance between the reverse process mean approximator and the forward process posterior mean. This could then be reparameterized such that the reverse process mean function approximator was solely responsible for predicting the added noise to the original image[Eq. 12]. The authors noted that the approximator could also predict the original image, but in their research, this led to lower-quality results. The parameterization of the function approximator to predicting added noise bore similarities to both Langevin dynamics and [denoising score-matching](https://arxiv.org/pdf/1907.05600).

The L<sub>0</sub> term could then be modeled using a separate, discrete decoder, although the authors left open the possibility that a more powerful, autoregressive decoder could also be employed. Having defined the reverse process and its final step decoder, the authors tweaked the final loss term, removing the weighting coefficient [Eq. 14], finding it improved sample quality and was easier to implement for model training. Additionally, the weighting coefficient from Eq. 12 down-weighted early steps in the Markov Chain, which was effectively implemented in the linear schedule of the forward process variance.

Supplementary conclusions from the paper determined that minimizing the early steps of the denoising process led to increased sample quality, likely as a result of allowing the model to focus on the more difficult denoising tasks. Researchers selected a U-Net architecture to model their training objective and observed its success at inference time despite non-competitive lossless codelengths, affirming the U-Net’s superior performance with spatial data. Other experimental details were maintained in later literature: temporal information was embedded into the U-Net, parameters were shared across timesteps, and each forward process in the noising process scaled down the signal input to prevent exploding variances across the chain. It also served to provide consistently scaled inputs to every subsequent latent in the trajectory. Similar to [previous](https://arxiv.org/pdf/1503.03585) [literature](https://arxiv.org/pdf/1907.05600), 1000 steps at inference time were taken to gauge sample quality. This also served as a baseline for future diffusion research. Researchers also examined the rate-distortion tradeoff of devoting bits to model images and found that a majority of bits were devoted to modeling imperceptible image details. This conclusion was a consistent issue in pixel-space diffusion and was the underlying theory behind [Latent Diffusion Models](https://arxiv.org/pdf/2112.10752) and their decision to first encode images prior to learning their conceptual composition. The original graph can be seen below.

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/DDPM_graph.png" width="45%" /> 
</p>

DDPMs were the first step taken to revisit diffusion theory for image generation and, contemporarily, performed competitively in many image quality metrics. They were reliant on an unconditional model architecture with a wholly stochastic generative process, thanks to the introduction of an additional noise component in the generative process. Image generation and training was also prohibitively expensive due to pixel-space interpolations, with training on the CIFAR-10 dataset requiring the equivalent of 84.8 hours on a V100 GPU. DDPMs served as an important step forward in the utilization of diffusion theory for deep generative models, where they quickly demonstrated impressive capacity along a variety of applications. 

## iDDPM

## DDIM

[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com) were directly inspired by DDPMs. Their main goal was to achieve a faster inference time than DDPMs without sacrificing sample quality, proving that diffusion models could be competitive with GANs and other image generation architectures. Their key insight was examining non-Markovian inference processes with an equivalent surrogate objective function to DDPMs. DDPMs depended on a Markov-like chain of denoising to reverse the addition of noise to an image. To prevent this sequential dependence on the previous step for inference, researchers posited a non-Markovian generation process for decreased latency. Below, the left side of the image represents a Markov chain for denoising used in the DDPM paper. The right side of the image contains a graphical illustration of the DDIM process, where the progressive denoising of an image is not wholly dependent on the previous stage.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/Markovian_vs_non_Markovian_inference.png" alt="Graphical illustrations of Markovian and non-Markovian inference processes taken from DDIM research paper" width="85%">
</p>

Researchers observed that the DDPM objective focused on the marginal probability of a denoised image given its noisy latent, and not on the joint probability distribution of all images in the denoising process given the noisy latent. However, multiple joint probability distributions exist with equivalent marginals. This takeaway expanded the possible generative processes that could be implemented. Authors of the DDPM paper had already leveraged the Gaussian natures of the added noise and denoising probability distributions to minimize their KL-divergence. By fixing the variance of these distributions, they expressed a relationship between each distribution's mean such that the addition of noise to the image was solely dependent on the previous level of noise. This allowed the noise process to be reversible, but slow. The DDIM paper expanded the additive noise process and introduced a vector sigma. The vector sigma controls the stochasticity of the inference process. Setting sigma to $`\sqrt{(1-α_{t-1}) / (1-α_{t})} \sqrt{1-(α_t / α_{t-1})}`$ for all t leads to the DDPM case. With sigma set to 0, researchers arrived at an entirely deterministic image generation process. Below, you can see the new additive noise distribution and the accompanying generalized sampling equation for denoising.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/DDIM_sampling_process.png" width="70%">
</p>

Regardless of the value of sigma, the training objective remains the same across the family of distributions. This allows for models previously trained with the original DDPM objective to interchangeably use different sigma values in their sampling process. This was one of the first steps towards entirely segmenting the scheduling process and algorithm from the model training while preserving performance. 

The authors' primary intention with DDIMs was to arrive at an algorithm with a faster inference speed. Utilizing a deterministic generation process allowed for that acceleration. After all if you know where you're going, why walk? Rather than iterating over every possible stage in the denoising sequence, they chose to jump subsets of these stages while maintaining the marginal distribution. A model previously requiring 1000 forward steps could now be sampled with a subset of 20 stages at each step. This smaller sampling trajectory would lead to only 50 necessary jumps along the generative process, but still maintain the same objective function and ultimate destination as the previous model that would have taken 1000 steps to arrive. A graphical concept of this illustration is provided below.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/accelerated_non_Markovian_inference.png" alt="Image of accelerated non-Markovian inference process taken from DDIM research paper" width="60%">
</p>

The authors recognized that having a deterministic process could also allow more influence over image interpolation. Images synthesized from the same latent variable would inevitably share high-level features. Linear combinations of latent variables would lead to a blend of each image concept in the final output. This afforded a cheaper interpolation space than DDPMs, along with more control and replicability over image results.

## Euler

If DDIMs laxly negotiated the loosening of constraints accompanying the DDPM theory, [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) arrived with a bang and a demand for further liberation from said constraints. The first sentence announced the authors' intent: "We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices". They succeeded. Over a third of the schedulers currently available (as of January 2024) on Hugging Face for Stable Diffusion implementation owe their theoretical background to this literature[2]. If you're looking for another perspective on this paper in particular, which influenced so many scheduling algorithms I highly recommend you check out this [video walkthrough](https://www.youtube.com/watch?v=T0Qxzf0eaio) which offers a digestible and intuitive understanding.

This paper begins by examining the theoretical backing of diffusion-based image generation models. The authors observe that, in an eagerness for solid theoretical footing, many researchers restricted the design space for image synthesis models to ideas running in parallel to established mathematical theory. Following similar literature, the authors begin by utilizing an [ordinary differential equation (ODE)](https://en.wikipedia.org/wiki/Ordinary_differential_equation) to describe the probability flow modeling the denoising process. The authors argue that most papers employ Euler's method, but that a higher-order solver actually leads to an increase in compute efficiency. Solving an ODE numerically inherently involves an approximation of the true solution. At each step, approximating the solution trajectory introduces truncation error, scaling superlinearly as a function of the step size. Taking smaller steps leads to a more accurate approximation, at the expense of calls to the denoising neural network. Researchers observed that these calls to the denoiser bore the greatest correlation to both time and compute resources in arriving at a denoised image. They propose a parametrization allowing for smaller steps along the trajectory approaching the denoised image, while taking larger steps at the onset of the denoising process.

The authors also argue that selecting a solver is often over-valued. Many components of image synthesis models (network architecture, training details, sampler algorithms) previously thought to be highly dependent on one another are modular, and theoretical constraints placed on sampling algorithms to align with the overall model can be loosened. To prove this, they plug-and-play their higher-order sampler with parametrized time-discretization steps into previous model architectures and compare against both the model baselines and another plug-and-play configuration with a scheduling and scaling approach equivalent to that taken in DDIMs. Lower [Frechet Inception Distance (FID)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) scores were reached with a plug-and-play approach, not requiring the networks' retraining and supporting the modularity of image synthesis models.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Schedulers_Images/euler_graphical_comparison.png" alt="Graphical illustration taken from Karras et. al 2022 research paper" width="85%">
</p>

Generalized ODE allows for directly specifying any sigma(t).

Next, the authors turned to the role of stochasticity in latent variable denoising. Ostensibly, ODEs and [stochastic differential equations (SDEs)](https://en.wikipedia.org/wiki/Stochastic_differential_equation) cover the same trajectory, but a stochastic approach to denoising introduces new noise at every timestep it removes noise. This stochastic behavior might nudge a generated image to better approximate the denoising trajectory, or it might wander from the optimal trajectory and lead to a lower-quality image. This examination of stochastic behavior and its working in concert with ODE solutions was the guiding intuition behind ancestral schedulers for Stable Diffusion models. Ancestral schedulers balance the navigation of the solution trajectory with forays into alternative branches of the image denoising probability distribution. They walk away from the purely deterministic approach of ODEs, and often generate new and unexplored image generation possibilities from the same latent noise selection.


Want to wrap input in signal management layers.
Talk about avoiding connection directly between noisy input to network and direct output from denoised image to final network layer. Backbone of LDMs! Instead have VAE overarching with diffusion model in the middle.

## PLMS

[PLMS](https://arxiv.org/abs/2202.09778?ref=blog.segmind.com)

# Conclusion

## Citations

[1] https://www.youtube.com/watch?v=IVcl0bW3C70&t=903s

[2] https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/euler
