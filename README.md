# Graph Convolutional Networks for Relational Link Prediction

This repository contains a TensorFlow implementation of Relational Graph Convolutional Networks (R-GCN), as well as experiments on relational link prediction. The description of the model and the results can be found in our paper:

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
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and {van den Berg}, Rianne and Titov, Ivan and Welling, Max},
  booktitle={The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3--7, 2018, Proceedings 15},
  pages={593--607},
  year={2018},
  organization={Springer}
}
```
