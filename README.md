# Graph Convolutional Networks for Relational Link Prediction


## COMP 550 changes
### TensorFlow 2 port 

We ported this project to use the TensorFlow 2 API. In particular, the following changes were made:
* Used rewrite script to generate TF2 code (see https://www.tensorflow.org/guide/migrate/upgrade)
* Deleted TF1 codebase
* TF2 code now is in `code-tf2/`
* Changed code in `encoders/affine_transform.py` to use TF2 API (search for "TF2 compatibility")
* Disabled eager execution for now (see top of `code-tf2/train.py`)

Example train script:
`python.exe code-tf2/train.py --settings ../settings/gcn_basis.exp --dataset ../data/FB-Toutanova`

***
***

This repository contains a TensorFlow implementation of Relational Graph Convolutional Networks (R-GCN), as well as experiments on relational link prediction. The description of the model and the results can be found in out paper:

[Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling (ArXiv 2017)

**Requirements**

* TensorFlow (1.4)

**Running demo**

We provide a bash script to run a demo of our code. In the folder *settings*, a collection of configuration files can be found. The block diagonal model used in our paper is represented through the configuration file *settings/gcn_block.exp*. To run a given experiment, execute our bash script as follows:

```
bash run-train.sh \[configuration\]
```

We advise that training can take up to several hours and require a significant amount of memory.

**Citation**

Please cite our paper if you use this code in your own work:

```
@article{schlichtkrull2017modeling,
  title={Modeling Relational Data with Graph Convolutional Networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  journal={arXiv preprint arXiv:1703.06103},
  year={2017}
}
```
