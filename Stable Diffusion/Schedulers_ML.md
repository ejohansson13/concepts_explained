# Schedulers

This page is intended to cover schedulers and the more popular scheduling algorithms utilized with Stable Diffusion models. This is intended to be a summary overview, and will focus on broad comparison between the research papers inspiring these algorithms. If you are looking for a detailed examination of the algorithms' theoretical intricacies, I would suggest looking elsewhere. In this page we will abstract some of these details, and summarize each paper's contribution to the serviceability of scheduling algorithms within latent diffusion models.

All quotes, ideas, and images from the summarization of each research paper are taken from the accompanying paper unless cited otherwise.

## DDPM

Every scheduling algorithm we will look at today was inspired by the [Denoising Diffusion Probabilistic Models (DDPM) paper](https://arxiv.org/abs/2006.11239), which examined diffusion models as Markov chains with Gaussian elements. When training, the paper optimized across the variational bound of the negative log likelihood. The reordering of the negative log likelihood provided the below equation. 

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/ddpm_nll_optimized.png" alt="Optimized negative log likehood equation for DDPM" width="100%">
</p>

The equation contains comparisons between Gaussians via [KL divergences](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). The authors recognized that this could be resolved with a [Rao-Blackwell](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) approach rather than [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimates. The benefit of Rao-Blackwell equations is their transformation of equations into estimators optimizable via mean-squared-error (MSE), a common quality metric within machine learning. This allowed each term within the log-likelihood to be optimized with respect to MSE, in contrast to Monte Carlo estimates which depend on repeated random sampling for numerical coherence. 

"Reweighted variational objective (DDPMS) undersamples initial denoising steps to avoid spending abundance of resources modeling imperceptible details." Essentially DDPMs defined the training objective FOR diffusion models, DDIMs broadened it, and Euler blew it open. Previously, diffusion models may have generated good results, but operated in pixel-space (DDIMs closer to latent space, Euler even closer?) and did not have an optimized training objective.

DDIMs and Euler still have expensive training cost, according to Latent Diffusion Models paper.

Tying the L<sub>T</sub> term to constants allowed the term's treatment as a constant during training. The paper's insight came from viewing L<sub>t-1</sub> as "equal to (one term of) the variational bound for the Langevin-like reverse process". This broadened the microscope on diffusion model equations and recognized them as parallel to differential equations for nonequilibrium thermodynamics. Researchers could approximate denoising score matching as a sampling chain with [Langevin-like](https://en.wikipedia.org/wiki/Langevin_dynamics) dynamics. They recognized that inference of diffusion models bear a resemblance to stochastic differential equations used for modeling molecular systems' dynamics; the same tricks used in simplifying thermodynamic differential equations could be borrowed for diffusion models, accelerating the success of diffusion models for image generation tasks.

Some other important insights from the DDPM paper include the highlighting of semantic vs perceptual compression, a phenomenon referred to in previous literature as conceptual compression. The below graphs may look familiar. The original (left), taken from the DDPM paper, highlighted the rate-distortion tradeoff when training on the CIFAR-10 test set, with the majority of bits exhausted on imperceptible distortions. It was also used in the [Latent Diffusion Models paper](https://arxiv.org/pdf/2112.10752.pdf) (right) to emphasize the almost segmented learning process of image synthesis models, supporting the utilization of autoencoders in concert with diffusion models. Additionally, the authors' progressive lossy decompression scheme was considered a generalized application of autoregressive decoding, which ultimately became a key component of the [original Dall-E](https://arxiv.org/pdf/2102.12092.pdf) and [Parti](https://arxiv.org/pdf/2206.10789.pdf) research papers.

<img src="/Stable Diffusion/Images/DDPM_graph.png" width="45%" /> <img src="/Stable Diffusion/Images/LDM_graph.png" width="45%" />

Like all novel discoveries, DDPMs came with flaws. They relied on an impractically large number of inference steps, with the paper demonstrating satisfactory image quality and diversity after about 1000 steps. These steps followed the reversal of the diffusion process and were required to be performed sequentially, negating the computational power of GPUs in performing operations in parallel. DDPMs lacked control, as the model was entirely predicated on stochastic generation. Images would likely follow themes of their training data, but researchers had no further control over the generation process. The same latent noise variable could be denoised to a variety of results resembling the training images. These latent variables were also relatively high dimensional, forcing image interpolation near pixel-space and greatly increasing compute costs.

## DDIM

[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com) were directly inspired by DDPMs. Their main goal was to achieve a faster inference time than DDPMs without sacrificing sample quality, proving that diffusion models could be competitive with GANs and other image generation architectures. Their key insight was examining non-Markovian inference processes with an equivalent surrogate objective function to DDPMs. DDPMs depended on a Markov-like chain of denoising to reverse the addition of noise to an image. To prevent this sequential dependence on the previous step for inference, researchers posited a non-Markovian generation process for decreased latency. Below, the left side of the image represents a Markov chain for denoising used in the DDPM paper. The right side of the image contains a graphical illustration of the DDIM process, where the progressive denoising of an image is not wholly dependent on the previous stage.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Markovian_vs_non_Markovian_inference.png" alt="Graphical illustrations of Markovian and non-Markovian inference processes taken from DDIM research paper" width="85%">
</p>

Researchers observed that the DDPM objective focused on the marginal probability of a denoised image given its noisy latent, and not on the joint probability distribution of all images in the denoising process given the noisy latent. However, multiple joint probability distributions exist with equivalent marginals. This takeaway expanded the possible generative processes that could be implemented. Authors of the DDPM paper had already leveraged the Gaussian natures of the added noise and denoising probability distributions to minimize their KL-divergence. By fixing the variance of these distributions, they expressed a relationship between each distribution's mean such that the addition of noise to the image was solely dependent on the previous level of noise. This allowed the noise process to be reversible, but slow. The DDIM paper expanded the additive noise process and introduced a vector sigma. The vector sigma controls the stochasticity of the inference process. Setting sigma to $`\sqrt{(1-α_{t-1}) / (1-α_{t})} \sqrt{1-(α_t / α_{t-1})}`$ for all t leads to the DDPM case. With sigma set to 0, researchers arrived at an entirely deterministic image generation process. Below, you can see the new additive noise distribution and the accompanying generalized sampling equation for denoising.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/DDIM_sampling_process.png" width="70%">
</p>

Regardless of the value of sigma, the training objective remains the same across the family of distributions. This allows for models previously trained with the original DDPM objective to interchangeably use different sigma values in their sampling process. This was one of the first steps towards entirely segmenting the scheduling process and algorithm from the model training while preserving performance. 

The authors' primary intention with DDIMs was to arrive at an algorithm with a faster inference speed. Utilizing a deterministic generation process allowed for that acceleration. After all if you know where you're going, why walk? Rather than iterating over every possible stage in the denoising sequence, they chose to jump subsets of these stages while maintaining the marginal distribution. A model previously requiring 1000 forward steps could now be sampled with a subset of 20 stages at each step. This smaller sampling trajectory would lead to only 50 necessary jumps along the generative process, but still maintain the same objective function and ultimate destination as the previous model that would have taken 1000 steps to arrive. A graphical concept of this illustration is provided below.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/accelerated_non_Markovian_inference.png" alt="Image of accelerated non-Markovian inference process taken from DDIM research paper" width="60%">
</p>

The authors recognized that having a deterministic process could also allow more influence over image interpolation. Images synthesized from the same latent variable would inevitably share high-level features. Linear combinations of latent variables would lead to a blend of each image concept in the final output. This afforded a cheaper interpolation space than DDPMs, along with more control and replicability over image results.

## Euler

If DDIMs laxly negotiated the loosening of constraints accompanying the DDPM theory, [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) arrived with a bang and a demand for further liberation from said constraints. The first sentence announced the authors' intent: "We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices". They succeeded. Over a third of the schedulers currently available (as of January 2024) on Hugging Face for Stable Diffusion implementation owe their theoretical background to this literature[2]. If you're looking for another perspective on this paper in particular, which influenced so many scheduling algorithms I highly recommend you check out this [video walkthrough](https://www.youtube.com/watch?v=T0Qxzf0eaio) which offers a digestible and intuitive understanding.

This paper begins by examining the theoretical backing of diffusion-based image generation models. The authors observe that, in an eagerness for solid theoretical footing, many researchers restricted the design space for image synthesis models to ideas running in parallel to established mathematical theory. Following similar literature, the authors begin by utilizing an [ordinary differential equation (ODE)](https://en.wikipedia.org/wiki/Ordinary_differential_equation) to describe the probability flow modeling the denoising process. The authors argue that most papers employ Euler's method, but that a higher-order solver actually leads to an increase in compute efficiency. Solving an ODE numerically inherently involves an approximation of the true solution. At each step, approximating the solution trajectory introduces truncation error, scaling superlinearly as a function of the step size. Taking smaller steps leads to a more accurate approximation, at the expense of calls to the denoising neural network. Researchers observed that these calls to the denoiser bore the greatest correlation to both time and compute resources in arriving at a denoised image. They propose a parametrization allowing for smaller steps along the trajectory approaching the denoised image, while taking larger steps at the onset of the denoising process.

The authors also argue that selecting a solver is often over-valued. Many components of image synthesis models (network architecture, training details, sampler algorithms) previously thought to be highly dependent on one another are modular, and theoretical constraints placed on sampling algorithms to align with the overall model can be loosened. To prove this, they plug-and-play their higher-order sampler with parametrized time-discretization steps into previous model architectures and compare against both the model baselines and another plug-and-play configuration with a scheduling and scaling approach equivalent to that taken in DDIMs. Lower [Frechet Inception Distance (FID)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) scores were reached with a plug-and-play approach, not requiring the networks' retraining and supporting the modularity of image synthesis models.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/euler_graphical_comparison.png" alt="Graphical illustration taken from Karras et. al 2022 research paper" width="85%">
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
