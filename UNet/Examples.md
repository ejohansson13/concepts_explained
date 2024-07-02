# Examples

Below are some examples of the U-Net's functionality from a self-trained U-Net on the following [dataset](https://molab.es/datasets-brain-metastasis-1/?type=metasrd). The dataset contains images of a metastasis in the brain from Patient 040102. More information can be found in the code subdirectory of the U-Net folder. The U-Net was provided high-resolution imaging of the patient's brain across multiple time points and slowly learned to segment the metastasis from the provided annotated segmentations before being evaluated on images it was not trained on.

<p align="center" width=100%>
  <img src="/UNet/Images/0172_img.png" width="15%" /> <img src="/UNet/Images/0172_msk.png" width="15%" /> <img src="/UNet/Images/0172_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0185_img.png" width="15%" /> <img src="/UNet/Images/0185_msk.png" width="15%" /> <img src="/UNet/Images/0185_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0205_img.png" width="15%" /> <img src="/UNet/Images/0205_msk.png" width="15%" /> <img src="/UNet/Images/0205_pred.png" width="15%" />
</p>

As you can see above, the model demonstrates some success in segmenting the larger instances in the brain, but lacks nuance. The provided ground-truth examples mirror a coastline, accounting for minute details in the metastasis area. The U-Net predictions lack this detail, and favor a circular segmentation, likely resulting from the loss metrics the model was trained on and the minimal resources put towards training this model. Let's look at how the model performs with smaller segmentation areas. Does the struggle to capture detail in the segmentation area result in an inability to segment smaller instances?
<p align="center" width=100%>
  <img src="/UNet/Images/0457_img.png" width="15%" /> <img src="/UNet/Images/0457_msk.png" width="15%" /> <img src="/UNet/Images/0457_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0530_img.png" width="15%" /> <img src="/UNet/Images/0530_msk.png" width="15%" /> <img src="/UNet/Images/0530_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0531_img.png" width="15%" /> <img src="/UNet/Images/0531_msk.png" width="15%" /> <img src="/UNet/Images/0531_pred.png" width="15%" />
</p>

It's a mixed bag. Some smaller segmentation instances are captured well by the model, while it misses others entirely. The U-Net still favors a circular segmentation area, regardless of the size, for all predictions. The U-Net has demonstrated success with large and small segmentation areas. Its primary limitation seems to be its inability to capture the nuance of segmentation instances. Does the U-Net demonstrate any further issues in its segmentation predictions?

<p align="center" width=100%>
  <img src="/UNet/Images/0551_img.png" width="15%" /> <img src="/UNet/Images/0551_msk.png" width="15%" /> <img src="/UNet/Images/0551_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/0552_img.png" width="15%" /> <img src="/UNet/Images/0552_msk.png" width="15%" /> <img src="/UNet/Images/0552_pred.png" width="15%" />
</p>

In both of the images above, the segmentation area is large. We might expect our model to provide an inadequate border of the area, similar to our previous examples. Instead, the model predicts a much smaller segmentation area. The model lacks confidence in predicting a larger segmentation area despite previously successful performances with similarly sized segmentation instances. Let's take a look through the lens of the U-Net and what the model receives as input to understand its decision-making. 

<p align="center" width=100%>
  <img src="/UNet/Images/z_0551_img.png" width="15%" /> <img src="/UNet/Images/0551_msk.png" width="15%" /> <img src="/UNet/Images/0551_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0552_img.png" width="15%" /> <img src="/UNet/Images/0552_msk.png" width="15%" /> <img src="/UNet/Images/0552_pred.png" width="15%" />
</p>

It's difficult to see any segmentation area and understandable why the model would struggle to correctly highlight the relevant area. Let's revisit our original three examples to view the correlation between model input and a more successful segmentation prediction.

<p align="center" width=100%>
  <img src="/UNet/Images/z_0172_img.png" width="15%" /> <img src="/UNet/Images/0172_msk.png" width="15%" /> <img src="/UNet/Images/0172_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0185_img.png" width="15%" /> <img src="/UNet/Images/0185_msk.png" width="15%" /> <img src="/UNet/Images/0185_pred.png" width="15%" />
</p>

<p align="center" width=100%>
  <img src="/UNet/Images/z_0205_img.png" width="15%" /> <img src="/UNet/Images/0205_msk.png" width="15%" /> <img src="/UNet/Images/0205_pred.png" width="15%" />
</p>

Medical imaging is a complex technology. Patient movement during examination and instrument calibration play important roles in the success of [medical scans](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5447676/). The examples above highlight the limitations of an automatic detection model dependent on low signal-to-noise ratio imaging inputs. This model was far from perfect, it was created solely to serve as an implementation example for the U-Net technology presented in this page. Developing a model for industry would require many more iterations, careful tuning, and, in such a high-risk domain, likely still be semi-automatic and reliant on human annotation for support. The model would certainly need to be improved, but so could the data input to the model.

## Improving Input Data

The industry shift to [data improvement over model improvement](https://www.enterpriseai.news/2021/10/08/ai-modeling-reinvented-its-time-to-shift-to-better-data-rather-than-just-building-better-models/) for machine learning represents a fundamental change in perspective. The early focus of machine learning models was to improve models' performance by tuning parameters, activation functions, and altering model architectures. Newer advancements in machine learning are increasingly focused on improving the model input for superior model output. [Dall-E 3](https://cdn.openai.com/papers/dall-e-3.pdf) focused on improved text captioning in the training data for greater prompt fidelity in the generated images. That paper theorized that previous text to image models' inability to holistically capture the sentiment of textual prompts arose from "noisy and inacurate image captions in the training dataset". They remedied the situation by training a custom image captioning model to improve textual captions of training images and highlighted the importance of data in achieving efficacious generative models. The lead author of that paper, James Betker, affirmed as much in his [personal blog](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/). Successful model architectures with non-detrimental hyperparameters trained for a long enough time converge to approximately equivalent representations of their dataset. At once this is exceedingly apparent and incredibly unintuitive. If the fundamental role of a model is to accurately represent its data, it makes sense that architecture, optimization, and various configurations become arbitrary. Successful models will successfully model their data. But, this requires a strong mentality shift from academic projects in machine learning where the same choices rendered arbitrary by generative model-level training are the cornerstones of successful training on smaller datasets. All of which is a lot of words to say that the successful training of smaller models, e.g. this U-Net trained on biomedical images, is predicated on configuration choices but, even more so on the data it is trained on.
