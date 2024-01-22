This page is intended to cover schedulers and the more popular scheduling algorithms utilized in Latent Diffusion Models. This is intended to be a summary overview, and will focus on broad comparison between algorithms. If you are looking for a detailed examination of theoretical intricacies within these differential equations, I would suggest looking elsewhere. In this page we will abstract these details, and observe schedulers through the lens of their serviceability as noise schedulers for latent diffusion models.

Summary. 
Every scheduling algorithm we will look at today was derived from the original [DDPM paper](https://arxiv.org/abs/2006.11239), which examined diffusion models as Markov chains with Gaussian elements. When training, the paper optimized across the variational bound of the negative log likelihood. The reordering of the negative log likelihood provided the below equation. 

<p align="center" width="100%">
  Caption if wanted <br>
  <img src="/Stable Diffusion/Images/ddpm_nll_optimized.png" alt="Optimized negative log likehood equation for DDPM" width="25%">
</p>

The above equation contains comparisons between Gaussians via [KL divergences](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). The authors recognized that this could be resolved with with [Rao-Blackwell](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) methodologies rather than [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimates. The benefit of Rao-Blackwell equations is their transformation of equations into estimators that can be optimized through mean-squared-error criteria, a common quality metric within machine learning. Monte Carlo estimates, in contrast, depend on repeated random sampling for coherent numerical results. 

Benefits. Created KL divergences for negative log-likelihood which were comparisons between Gaussians, calculated with [Rao-Blackwell](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) methodology instead of Monte Carlo estimates. Lt is constant during training. Reverse process for one term in NLL (Lt-1) "equal to one term of variational bound for Langevin-like reverse process". Optimizing denoising score-matching is equivalent to using variational inference to fit finite-time [marginal](https://en.wikipedia.org/wiki/Marginal_distribution) of sampling chain with [Langevin-like dynamics](https://en.wikipedia.org/wiki/Langevin_dynamics). All of which is a lot of words to say, they recognized that inference of diffusion models beared a lot of resemblance to other stochastic differential equations used for modeling molecular systems' dynamics. 

Recognized semantic vs perceptual compression tradeoff. Graph from SD paper originally came from DDPM paper (include here). Progressive lossy decompression scheme resembles autoregressive decoding which became backbone of Dall-E 1 (and Imagen or Parti?). 

Drawbacks. Impractically large number of inference steps. Must be performed sequentially, can't be performed in parallel. Cannot encode samples from latent code due to stochastic sampling. No separation of scheduling algorithm from remainder of architecture. 

Something about how all schedulers recognized the calculus backbone of diffusion algorithms. Recognized that some of those details could be abstracted away, Euler by Nvidia took it to another level. 

[DDIM](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com)

[Euler](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) 
