# Logo Image Composition with Generative Diffusion Models

Generative diffusion models have demonstrated significant success in image composition, particularly for integrating naturalistic images with corresponding backgrounds. However, the specific challenge of composing logos—often visually distinct from typical backgrounds—has not been as thoroughly explored.

In this project, we propose a novel approach that utilizes generative diffusion models for logo composition. Our method employs a conditioned diffusion model to seamlessly integrate logos into various background images. To support our research, we constructed a custom dataset consisting of 120 pairs of logos and background images.

We evaluate the performance of our model using two key metrics: Frechet Inception Distance (FID) and CLIP Similarity Score. Our findings indicate that while our model can effectively compose logos onto diverse backgrounds, it encounters challenges in preserving the structural integrity of logos that feature text or intricate patterns.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ce8583b-1912-42d6-b41e-b84d4d7b1ce2" width="50%" hspace="10"/>
</p>

## References

References
- [1] Roy Hachnochi, Mingrui Zhao, Nadav Orzech, Rinon Gal, Ali Mahdavi-Amiri, Daniel Cohen-Or, and Amit Haim
Bermano. Cross-domain compositing with pretrained diffusion models, 2023.
- [2] Divya Bhargavi, Karan Sindwani, and Sia Gholami. Zero-shot virtual product placement in videos. In
Proceedings of the 2023 ACM International Conference on Interactive Media Experiences, IMX ’23, page
289–297, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400700286. doi:
10.1145/3573381.3597213. URL https://doi.org/10.1145/3573381.3597213.
- [3] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen.
Paint by example: Exemplar-based image editing with diffusion models. In 2023 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 18381–18391, 2023. doi: 10.1109/CVPR52729.
2023.01763.
- [4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image
synthesis with latent diffusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 10674–10685, 2022. doi: 10.1109/CVPR52688.2022.01042.
- [5] Andras Tüzkö, Christian Herrmann, Daniel Manger, and Jürgen Beyerer. Open set logo detection and retrieval. In VISIGRAPP, 2017. URL https://api.semanticscholar.org/CorpusID:4400394.
- [6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_
files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf.
- [7] D.C Dowson and B.V Landau. The fréchet distance between multivariate normal distributions. Journal of
Multivariate Analysis, 12(3):450–455, 1982. ISSN 0047-259X. doi: https://doi.org/10.1016/0047-259X(82)
90077-X. URL https://www.sciencedirect.com/science/article/pii/0047259X8290077X.
- [8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning
transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021. URL https:
//arxiv.org/abs/2103.00020.
- [9] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona,
Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015.
- [10] Jing Wang, Weiqing Min, Sujuan Hou, Shengnan Ma, Yuanjie Zheng, and Shuqiang Jiang. Logodet-3k: A
large-scale image dataset for logo detection. ACM Trans. Multimedia Comput. Commun. Appl., 18(1), jan 2022. ISSN 1551-6857. doi: 10.1145/3466780. URL https://doi.org/10.1145/3466780.
- [11] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks, 2023.
Gantugs Atarsaikhan, Brian Kenji Iwana, and Seiichi Uchida. Constrained neural style transfer for decorated
logo generation, 2018.
- [12] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations,
ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http:
//arxiv.org/abs/1409.1556.
