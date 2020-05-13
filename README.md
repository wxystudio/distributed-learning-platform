# distributed learning
a distributed learning method based on this paper:privacy preserving machine learning
https://www.comp.nus.edu.sg/~reza/files/Shokri-CCS2015.pdf
this is a platform used for distributed training, users with di erent dataset can train a general neural network. This model can be more powerful than users training alone. It can also protect users’ privacy.

这是我之前的paper(https://arxiv.org/abs/1911.08128) 所对应的code，基于上面CCS 15的paper给我的灵感，我一共用了三种手段实现了分布式生成对抗网络，可以做到每个用户都掌握一部分数据，而最终训练出一个模型，可以生成所有的数据，我们认为这有助于不愿意上传自己数据的用户保护自己的真实数据，并且训练出自己想要的模型。举一个例子，比如MNIST数据集，A用户拥有数据0-4，B用户拥有数据5-9，那么我们可以通过这个平台，让他们在本地训练的情况下，不上传自己的真实数据，训练出一个GAN模型，生成0-9的数据，这样就会增加用户的安全感。
