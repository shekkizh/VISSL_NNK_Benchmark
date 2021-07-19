# VISSL_NNK_Benchmark
NNK evaluation of Self supervised models. The source code is integrated with VISSL for feature extraction and can be applied to any dataset and model that can be loaded by VISSL. 
One can also, perform evaluation on pre

# Requirements
The source code assumes the following packages are installed: 
- VISSL: This forms the bacbone for retrieving features of various self supervised models. The code was tested with installation from VISSL source.
- FAISS: Used to query for nearest neighbors. Code was tested with pip installed gpu version.
Installation instructions for both packages can be found [here](https://github.com/facebookresearch/vissl/blob/master/INSTALL.md) and [here](Installation instructions can be found [here](https://github.com/facebookresearch/vissl/blob/master/INSTALL.md))

# Data and Pretrained models
The config files provided assume evaluation on ImageNet data which need to be downloaded and saved to a local directory. Further instructions on how to setup the dataset
can be found in VISSL [documentation](https://vissl.readthedocs.io/en/v0.1.5/getting_started.html#setup-dataset).

VISSL documentation also provides [instruction](https://vissl.readthedocs.io/en/v0.1.5/evaluations/feature_extraction.html) and tutorials(https://github.com/facebookresearch/vissl#tutorials) on loading pretrained self supervised models for feature extraction.

