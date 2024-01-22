This page is intended to cover schedulers and the more popular scheduling algorithms utilized in Latent Diffusion Models. This is intended to be a high-level overview, and will focus on comparison of latency, parallelism, and the "spirit" of the algorithm. If you are looking for an exhaustive examination of the theoretical intricacies of these differential equations from a mathematical perspective, I would suggest looking elsewhere. In this page we will abstract these details, and observe schedulers through the lens of their serviceability as noise schedulers for latent diffusion models.

Summary. 
Every scheduling algorithm we will look at today was derived from the original [DDPM paper](https://arxiv.org/abs/2006.11239), which examined diffusion models as Markov chains with Gaussian elements. 

Benefits. 

Drawbacks. 

Something about how all schedulers recognized the calculus backbone of diffusion algorithms. Recognized that some of those details could be abstracted away, Euler by Nvidia took it to another level. 

[DDIM](https://arxiv.org/abs/2010.02502?ref=blog.segmind.com)

[Euler](https://arxiv.org/abs/2206.00364?ref=blog.segmind.com) 
