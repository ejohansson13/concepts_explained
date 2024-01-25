# Schedulers

This page is intended to cover schedulers and the more popular scheduling algorithms utilized in Latent Diffusion Models. This is intended to be a summary overview, and will focus on broad comparison between algorithms. If you are looking for a detailed examination of theoretical intricacies within these differential equations, I would suggest looking elsewhere. In this page we will abstract these details, and observe schedulers through the lens of their serviceability as noise schedulers for latent diffusion models.

## DDPM

Every scheduling algorithm we will look at today was derived from the original [Denoising Diffusion Probabilistic Models (DDPM) paper](https://arxiv.org/abs/2006.11239), which examined diffusion models as Markov chains with Gaussian elements. When training, the paper optimized across the variational bound of the negative log likelihood. The reordering of the negative log likelihood provided the below equation. 

<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/ddpm_nll_optimized.png" alt="Optimized negative log likehood equation for DDPM" width="100%">
</p>

The above equation contains comparisons between Gaussians via [KL divergences](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). The authors recognized that this could be resolved with with [Rao-Blackwell](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) methodologies rather than [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimates. The benefit of Rao-Blackwell equations is their transformation of equations into estimators that can be optimized through mean-squared-error criteria, a common quality metric within machine learning. Monte Carlo estimates, in contrast, depend on repeated random sampling for coherent numerical results. By treating the negative log likelihood as a summation of Rao-Blackwell equations, each term within the log likelihood could be optimized with respect to mean-squared-error. Tying the L<sub>T</sub> term to constants allowed the term's treatment as a constant during training. The great insight of the paper came from viewing L<sub>t-1</sub> as "equal to (one term of) the variational bound for the Langevin-like reverse process". This broadened the microscope on diffusion model equations and recognized them as parallel to differential equations for nonequilibrium thermodynamics. Researchers could approximate denoising score matching as a sampling chain with [Langevin-like](https://en.wikipedia.org/wiki/Langevin_dynamics) dynamics. All of which is a lot of words to say, they recognized that inference of diffusion models beared a lot of resemblance to other stochastic differential equations used for modeling molecular systems' dynamics. Tricks used in simplifying thermodynamic differential equations could be borrowed for diffusion models, a conclusion that influenced subsequent scheduling algorithms and accelerated the success of diffusion models for image generation tasks.

Some other important insights from the DDPM paper include the highlighting of semantic vs perceptual compression, a phenomenon referred to in previous literature as conceptual compression. The below graphs may look familiar. The original (left), taken from the DDPM paper, highlighted the rate-distortion tradeoff when training on the CIFAR-10 test set, with the majority of bits exhausted on imperceptible distortions. It was also used in the [Latent Diffusion Models paper](https://arxiv.org/pdf/2112.10752.pdf) (right) to emphasize the almost segmented learning process of image synthesis models, supporting the utilization of autoencoders in concert with diffusion models. Additionally, the authors' progressive lossy decompression scheme was considered a generalized application of autoregressive decoding, which ultimately became a key component of the [original Dall-E](https://arxiv.org/pdf/2102.12092.pdf) and [Parti](https://arxiv.org/pdf/2206.10789.pdf) research papers.

<img src="/Stable Diffusion/Images/DDPM_graph.png" width="45%" /> <img src="/Stable Diffusion/Images/LDM_graph.png" width="45%" />

Unfortunately, like all novel discoveries, it came with flaws. DDPMs relied on an impractically large number of inference steps, with the paper only demonstrating satisfactory image quality and diversity after about 1000 steps. These steps followed the reversal of the diffusion process and were required to be performed sequentially, negating the computational power of GPUs in performing operations in parallel. Lastly, the model was unable to encode samples from the latent space for a visualization of the generation process due to the stochastic sampling of its diffusion process.

Also not deterministic because of stochasticity. Interpolate near image space due to stochastic generative process. Latent variables are relatively high dimensional.

## DDIM

[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com) were directly inspired by DDPMs. Their main goal was to achieve a faster inference time than DDPMs without sacrificing sample quality, proving that diffusion models could be competitive with GANs and other image generation architectures. Their key insight was examining non-Markovian inference processes with an equivalent surrogate objective function to DDPMs. DDPMs depended on a Markov-like chain of denoising reversing the learned addition of noise to image. To prevent this sequential dependence on the previous step for inference, researchers posited a non-Markovian generation process for decreased latency.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/Markovian_vs_non_Markovian_inference.png" alt="Graphical illustrations of Markovian and non-Markovian inference processes taken from DDIM research paper" width="60%">
</p>

Researchers found that the DDPM objective focused on the marginal probability distribution of a noisy image given the denoised image, and not directly on the joint probability distribution of all noisy images given the denoised image. Multiple joint probability distributions can exist with equivalent marginal probability distributions. This takeaway expanded the possible generative processes that could be implemented. Authors of the DDPM paper had already leveraged the Gaussian nature of the added noise probability distribution to minimize the KL-divergence between the added noise probability distribution and the denoising distribution. By fixing the variance of these distributions, they had expressed a relationship between the means of each distribution such that each addition of noise to the image was solely dependent on the previous image's level of noise. This made reversing the noise process simplistic, albeit slow. The DDIM paper actually expanded the additive noise process and introduced a vector sigma (still maintained its Gaussian properties, but this is dismissable and an aside in the research paper). Regardless of the sigma value, the training objective remains the same. In fact, the same substitution utilized in the DDPM paper for sampling can be employed here to give our new sampling equation for denoising images.

The new "family of generative processes" are indexed by a vector sigma, and actually include the DDPM inference process as a special case. The vector sigma controls the stochasticity of the inference process. With sigma set to 0, researchers arrived at an entirely deterministic image generation process.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/DDIM_sampling_process.png" width="50%">
</p>


Now, cover inference speed up with DDIMs in comparison to DDPMs.
Controlling steps by traversing subsets of steps at each moment.
<p align="center" width="100%">
  <img src="/Stable Diffusion/Images/accelerated_non_Markovian_inference.png" alt="Image of accelerated non-Markovian inference process taken from DDIM research paper" width="60%">
</p>

Benefits.
Deterministic image output -> same latent will give you same image. Allowed for separation of scheduling algorithm from model architecture. "From the definition of Jσ, it would appear that a different model has to be trained for every choice of σ, since it corresponds to a different variational objective (and a different generative process). However, Jσ is equivalent to Lγ for certain weights γ, as we show below... With L1 as the objective, we are not only learning a generative process for the Markovian inference process considered in Sohl-Dickstein et al. (2015) and Ho et al. (2020), but also generative processes for many non-Markovian forward processes parametrized by σ that we have described. Therefore, we can essentially use pretrained DDPM models as the solutions to the new objectives, and focus on finding a generative process that is better at producing samples subject to our needs by changing σ."
Significantly improved inference time. Allowed tradeoff between inference speed and quality. Alteration of noise in latent space could controllably alter output of image in pixel space. 

Drawbacks.

## Euler

[Euler](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) 

## PLMS

[PLMS](https://arxiv.org/abs/2202.09778?ref=blog.segmind.com)

# Conclusion
