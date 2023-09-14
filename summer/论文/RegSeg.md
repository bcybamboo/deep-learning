[Rethink Dilated Convolution for Real-time Semantic Segmentation](https://github.com/RolandGao/RegSeg)

问题：绝大部分计算量集中在骨干网络中，而大部分骨干网络并不能获取足够大的感受野

部分解决方法：快速降低骨干网络中的分辨率来解决这一点，并且同时使用一个或多个具有高分辨率的平行分支（比如BiSeNet、STDC、DDRNet)

本文使用了不同的方法，设计一个受ResNeXt启发的结构，使用两个平行的3×3卷积，每个卷积拥有不同的dilation rate，扩大感受野的同时也能保留局部的细节特征。

还提出了一个轻量级的解码器，它比普通的替代方案能恢复更多的局部信息。

