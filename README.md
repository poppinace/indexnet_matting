# IndexNet Matting
This repository includes the official implementation of IndexNet Matting for deep image matting, presented in our paper:

**Indices Matter: Learning to Index for Deep Image Matting**

Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2019

[Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>, Yutong Dai<sup>1</sup>, [Chunhua Shen](http://cs.adelaide.edu.au/~chhshen/)<sup>1</sup>, Songcen Xu<sup>2</sup>

<sup>1</sup>The University of Adelaide, Australia

<sup>2</sup>Noah's Ark Lab, Huawei Technologies

## Highlights
- **Simple and effective:** IndexNet Matting only deals with the upsampling stage but exhibits at least 16.1% relative improvements, compared to the Deep Matting baseline;
- **Memory-efficient:** IndexNet Matting builds upon MobileNetV2. It can process an image with a resolution up to 1980x1080 on a single GTX 1070;
- **Compatible:** This framework also includes our re-implementation of DeepMatting and the pretrained model presented in the Adobe's CVPR17 paper.

## Installation
Our code has been tested on Python 3.6.8/3.7.2 and PyTorch 0.4.1/1.1.0. Please follow the official instructions to configure the environment. See other required packages in `requirements.txt`.

## A Quick Demo
We have included our pretrained model in `./pretrained` and a testing image and a trimap from the Adobe Image Dataset in `./examples`. Run the following command for a quick demonstration of IndexNet Matting. The inferred alpha matte is in the folder `./examples/mattes`.

    python scripts/demo.py
    
## Prepare Your Data
1. Please contact Brian Price (bprice@adobe.com) requesting for the Adobe Image Matting dataset;
2. Composite the dataset using provided foreground images, alpha mattes, and background images from the COCO and Pascal VOC datasets. I slightly modified the provided `compositon_code.py` to improve the efficiency, included in the `scripts` folder. Note that, since the image resolution is quite high, the dataset will be over 100 GB after composition.
3. The final path structure used in my code looks like this:


```
-->Combined_Dataset
    -->Training_set
        -->alpha (431 images)
        -->fg (431 images)
        -->merged (43100 images)
    -->Test_set
        -->alpha (50 images)
        -->fg (50 images)
        -->merged (1000 images)
        -->trimaps (1000 images)
```
## Code will be coming soon!

## References
```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

