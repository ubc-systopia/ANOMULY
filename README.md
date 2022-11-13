# Anomaly Detection in Multiplex Dynamic Networks: from Blockchain Security to Brain Disease Prediction 
ANOMULY is a general, unsupervised edge anomaly detection framework for multiplex dynamic networks.



#### Authors: [Ali Behrouz](https://abehrouz.github.io/), [Margo Seltzer](https://www.seltzer.com/margo/)
#### [Link to the paper](https://openreview.net/forum?id=UDGZDfwmay)
#### [Poster]()
#### [Brief video explanation]()
#### [Spotlight talk]()





### Abstract
The problem of identifying anomalies in dynamic networks is a fundamental task with a wide range of applications. However, it raises critical challenges due to the complex nature of anomalies,  lack of ground truth knowledge, and complex and dynamic interactions in the network. Most existing approaches usually study networks with a single type of connection between vertices, while in many applications interactions between objects vary, yielding multiplex networks. We propose ANOMULY, a general, unsupervised edge anomaly detection framework for multiplex dynamic networks. In each relation type, ANOMULY sees node embeddings at different GNN layers as hierarchical node states and employs a GRU cell to capture temporal properties of the network and update node embeddings over time. We then add an attention mechanism that incorporates information across different types of relations. Our case study on brain networks shows how this approach could be employed as a new tool to understand abnormal brain activity that might reveal a brain disease or disorder. Extensive experiments on nine real-world datasets demonstrate that ANOMULY achieves state-of-the-art performance.



### Datasets
Links to datasets used in the paper:
RM: https://github.com/joint-em/FTCS   
Brain Networks: http://umcd.humanconnectomeproject.org/umcd/default/browse_studies  
DBLP and Amazon: https://github.com/joint-em/firmcore  
Ethereum and Ripple: https://github.com/tdagraphs  


### Dataset format





### Reference

```
@article{anomuly2022,
  title={Anomaly Detection in Multiplex Dynamic Networks: from Blockchain Security to Brain Disease Prediction},
  author={Ali Behrouz, Margo Seltzer},
  journal={NeurIPS Temporal Graph Learning Workshop},
  year={2022},
}
```
