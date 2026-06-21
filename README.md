# KGEs from RotatE
**Introduction**

A clone of RotatE's repo to test its KGEs. Thanks RotatE and its 'parents'.

**Implemented features**

Models:
 - [x] ComplEx

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Setup**

1. Create a virtual environment and activate
```
python -m venv .venv
```

2. Activate venv
```
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

**Train**

For example, this command train a RotatE model on FB15k dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --epochs 159 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

To reprocude the results in the ICLR 2019 paper [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ), you can run the bash commands in best_config.sh to get the best performance of RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Countries).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 212 16 -de

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.

Add your own model to model.py like:
```
def TransE(self, head, relation, tail, mode):
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = self.gamma.item() - torch.norm(score, p=1, dim=2)
    return score
```

**Citation**

If you use the codes, please cite the following [paper](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
