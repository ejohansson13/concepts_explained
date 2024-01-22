This page is intended to cover schedulers and the more popular scheduling algorithms utilized in Latent Diffusion Models. This is intended to be a high-level overview, and will focus on comparison of latency, parallelism, and the "spirit" of the algorithm. If you are looking for an exhaustive examination of the theoretical intricacies of these differential equations from a mathematical perspective, I would suggest looking elsewhere. In this page we will abstract these details, and observe schedulers through the lens of their serviceability as noise schedulers for latent diffusion models.

Summary. 
Every scheduling algorithm we will look at today was derived from the original [DDPM paper](https://arxiv.org/abs/2006.11239), which examined diffusion models as Markov chains with Gaussian elements. When training, the paper followed conventional wisdom, optimizing across the variational bound of the negative log likelihood. 

Benefits. Created KL divergences for negative log-likelihood which were comparisons between Gaussians, calculated with [Rao-Blackwell](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) methodology instead of Monte Carlo estimates. Lt is constant during training. Reverse process for one term in NLL (Lt-1) "equal to one term of variational bound for Langevin-like reverse process". Optimizing denoising score-matching is equivalent to using variational inference to fit finite-time [marginal](https://en.wikipedia.org/wiki/Marginal_distribution) of sampling chain with [Langevin-like dynamics](https://en.wikipedia.org/wiki/Langevin_dynamics). All of which is a lot of words to say, they recognized that inference of diffusion models beared a lot of resemblance to other stochastic differential equations used for modeling molecular systems' dynamics. 

Recognized semantic vs perceptual compression tradeoff. Graph from SD paper originally came from DDPM paper (include here). Progressive lossy decompression scheme resembles autoregressive decoding which became backbone of Dall-E 1 (and Imagen or Parti?). 

Drawbacks. Impractically large number of inference steps. Must be performed sequentially, can't be performed in parallel. Cannot encode samples from latent code due to stochastic sampling. No separation of scheduling algorithm from remainder of architecture. 

Something about how all schedulers recognized the calculus backbone of diffusion algorithms. Recognized that some of those details could be abstracted away, Euler by Nvidia took it to another level. 

[DDIM](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com)

[Euler](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) 
