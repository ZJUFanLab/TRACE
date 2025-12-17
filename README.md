![](https://img.shields.io/badge/version-1.0.0-blue)

# Transformer-based graphs for drug-drug interaction with chemical knowledge embedding

This repository is the official implementation of TRACE.

## Brief introduction

We propose TRACE, a new drug-drug interaction (DDI) prediction method, termed **TR**ansformer-based Graph Representation Le**A**rning with **C**hemical **E**mbedding.

### Model

During the construction of molecular graphs, we embed the elemental information from ElementKG into the original drug molecular graphs, resulting in KG-enhanced molecular graphs that combine both structural features and chemical domain knowledge.

After obtaining the KG-enhanced molecular graphs, we input them into the Graph Transformer module to further extract high-level representations of drug molecules. The Graph Transformer leverages self-attention mechanisms to dynamically aggregate local and global information within the graph, effectively capturing complex structural and chemical patterns critical for DDI prediction. The representations of two drug molecules are then concatenated and fed into a downstream multilayer perceptron (MLP), which is trained to predict DDIs.

![overview.png](overview.png)

## Requirements

To run our code, please install dependency packages.

```
python                       3.8
torch                        1.13.1
torch-scatter                2.0.9
rdkit                        2022.9.5
numpy                        1.24.3
dgl                          1.1.2+cu117
scikit-learn                 1.3.2
```

## Quick start

Run `graphpreprocess.py` to generate the molecular graph data. The output will be saved to the path specified by `data_bin_dir` in the YAML configuration file.

```bash
>> python graphpreprocess.py --config_file configs/config_deepddi.yml
```

```bash
>> python train_task.py --config_file configs/config_deepddi.yml
```

| **Parameter** | **Description** | **Example/Default Value** |
| --- | --- | --- |
| data_dir | Path to downstream task data file (.csv) | None |
| data_bin_dir | Path to graph data file (.pth) | None |
| id_to_index_dir | Path to drug ID and index mapping file (.csv) | None |
| best_model_dir | Path to save the best model | None |
| experiment_name | Name of the experiment | None |
| in_feats | Input feature dimension | 196 |
| in_edge_feats | Edge feature dimension | 6 |
| hidden_size | Hidden layer size | 128 |
| num_layers | Number of layers (e.g., Transformer/GCN layers) | 8 |
| num_class | Number of output classes | 2 |
| mlp_activation | Activation function used in MLP | silu |
| mlp_dropout_rate | Dropout rate in MLP | 0.5 |
| mlp_num_layers | Number of layers in MLP | 2 |
| mlp_batch_norm | Whether to use batch normalization in MLP | false |
| mlp_hidden_size | Hidden size in MLP | 2048 |
| short_cut | Whether to use shortcut connections | false |
| seed | Random seed | 123 |
| num_epochs | Number of training epochs | 5000 |
| batch_size | Batch size | 512 |
| learning_rate | Learning rate | 0.001 |
| checkpoint_dir | Directory to save checkpoints/outputs | ./output/ |
| patience | Patience for early stopping | 20 |
| early_stop_criteria | Early stopping evaluation metric | macro avg_f1-score |
| early_stop_mode | Early stopping mode (higher/lower) | higher |
| gpu | Whether to use GPU | true |
| criterion | Loss function | BCELoss |
| class_weight_dir | Path to class weights file (if any) | null |
| label_smoothing | Label smoothing factor | 0.1 |

## Dataset

- Download Drug-Drug Interaction dataset from https://github.com/isjakewong/MIRACLE/tree/main/MIRACLE/datachem.
    - Since these datasets include duplicate instances in train/validation/test split, merge the train/validation/test dataset.
    - Generate random negative counterparts by sampling a complement set of positive drug pairs as negatives.
    - Split the dataset into 6:2:2 ratio, and create separate csv file for each train/validation/test splits.

## Representative high-risk molecular motifs and drugs identified by TRACE for different DDI types
For each DDI type, we report the top-ranked molecular motifs identified by attention-based analysis in TRACE. Pair Count denotes the number of DDI pairs in which the motif appears among the top-ranked substructures of either drug, and Pair Coverage (%) represents the proportion of such pairs relative to all DDI pairs of that type. Example drugs are selected as the most frequently occurring drugs containing the corresponding motif within the given DDI type, and their molecular structures are shown for interpretability.

| DDI Type | Motif (SMILES + Structure) | Pair Count | Pair Coverage (%) | Representative Drug ID(s) | Drug (Structure) |
| --- | :-: | --- | --- | --- | :-: |
| 1 | C1Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3<br><img src="motifsvgg/motif_10271.svg" style="zoom:60%;" /> | 11 | 100.0 | DB00460 | <img src="smilessvg/DB00460.svg" style="zoom:50%;" /> |
| 1 | c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /> | 2 | 18.18 | DB04571<br>DB00553 | <img src="smilessvg/DB04571.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00553.svg" style="zoom:50%;" /> |
| 1 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 1 | 9.09 | DB01600 | <img src="smilessvg/DB01600.svg" style="zoom:50%;" /> |
| 2 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 59 | 18.27 | DB01409<br>DB06153 | <img src="smilessvg/DB01409.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06153.svg" style="zoom:50%;" /> |
| 2 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 48 | 14.86 | DB06148<br>DB00670 | <img src="smilessvg/DB06148.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00670.svg" style="zoom:50%;" /> |
| 2 | C1C[N+]2CCC1CC2<br><img src="motifsvgg/motif_4789.svg" style="zoom:60%;" /> | 45 | 13.93 | DB09076 | <img src="smilessvg/DB09076.svg" style="zoom:50%;" /> |
| 3 | c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /> | 11 | 2.12 | DB08906 | <img src="smilessvg/DB08906.svg" style="zoom:50%;" /> |
| 3 | C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /> | 11 | 2.12 | DB00288 | <img src="smilessvg/DB00288.svg" style="zoom:50%;" /> |
| 3 | C1CCCC1<br><img src="motifsvgg/motif_4874.svg" style="zoom:60%;" /> | 319 | 61.46 | DB04574<br>DB00588<br>DB08970 | <img src="smilessvg/DB04574.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00588.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08970.svg" style="zoom:50%;" /> |
| 4 | C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /> | 1893 | 37.78 | DB00794<br>DB01174<br>DB00312 | <img src="smilessvg/DB00794.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01174.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00312.svg" style="zoom:50%;" /> |
| 4 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 775 | 15.47 | DB00238<br>DB09280<br>DB01220 | <img src="smilessvg/DB00238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01220.svg" style="zoom:50%;" /> |
| 4 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 589 | 11.75 | DB00615<br>DB01074<br>DB11614 | <img src="smilessvg/DB00615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01074.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11614.svg" style="zoom:50%;" /> |
| 5 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 144 | 46.6 | DB00346<br>DB01162<br>DB00457 | <img src="smilessvg/DB00346.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00457.svg" style="zoom:50%;" /> |
| 5 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 105 | 33.98 | DB01162<br>DB00457<br>DB00590 | <img src="smilessvg/DB01162.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /> |
| 5 | c1cOCCO1<br><img src="motifsvgg/motif_3579.svg" style="zoom:60%;" /> | 34 | 11.0 | DB00590 | <img src="smilessvg/DB00590.svg" style="zoom:50%;" /> |
| 6 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 331 | 10.47 | DB09075<br>DB05266<br>DB00235 | <img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB05266.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00235.svg" style="zoom:50%;" /> |
| 6 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 263 | 8.32 | DB06228<br>DB06209<br>DB00758 | <img src="smilessvg/DB06228.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00758.svg" style="zoom:50%;" /> |
| 6 | c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /> | 187 | 5.92 | DB09075<br>DB01254<br>DB00814 | <img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00814.svg" style="zoom:50%;" /> |
| 7 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 6 | 28.57 | DB03615<br>DB01172<br>DB01421 | <img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /> |
| 7 | C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /> | 2 | 9.52 | DB01421<br>DB00452 | <img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /> |
| 7 | C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /> | 1 | 4.76 | DB00955 | <img src="smilessvg/DB00955.svg" style="zoom:50%;" /> |
| 8 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 26 | 10.61 | DB00295<br>DB01466<br>DB00844 | <img src="smilessvg/DB00295.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /> |
| 8 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 23 | 9.39 | DB01100<br>DB00297<br>DB01501 | <img src="smilessvg/DB01100.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00297.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01501.svg" style="zoom:50%;" /> |
| 8 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 23 | 9.39 | DB00363<br>DB04908<br>DB00875 | <img src="smilessvg/DB00363.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /> |
| 9 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 329 | 15.6 | DB01132<br>DB00468<br>DB00779 | <img src="smilessvg/DB01132.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00779.svg" style="zoom:50%;" /> |
| 9 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 148 | 7.02 | DB00359<br>DB06203<br>DB08882 | <img src="smilessvg/DB00359.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06203.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08882.svg" style="zoom:50%;" /> |
| 9 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 148 | 7.02 | DB01268<br>DB01200<br>DB06791 | <img src="smilessvg/DB01268.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06791.svg" style="zoom:50%;" /> |
| 10 | C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /> | 133 | 21.14 | DB00484<br>DB00575<br>DB06694 | <img src="smilessvg/DB00484.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00575.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06694.svg" style="zoom:50%;" /> |
| 10 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 126 | 20.03 | DB06237<br>DB06403<br>DB09242 | <img src="smilessvg/DB06237.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06403.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09242.svg" style="zoom:50%;" /> |
| 10 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 118 | 18.76 | DB00820<br>DB00206<br>DB01089 | <img src="smilessvg/DB00820.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00206.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01089.svg" style="zoom:50%;" /> |
| 11 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 57 | 18.21 | DB00758<br>DB06209<br>DB00208 | <img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00208.svg" style="zoom:50%;" /> |
| 11 | C1CNPOC1<br><img src="motifsvgg/motif_10011.svg" style="zoom:60%;" /> | 49 | 15.65 | DB01181 | <img src="smilessvg/DB01181.svg" style="zoom:50%;" /> |
| 11 | c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /> | 38 | 12.14 | DB00758<br>DB00208 | <img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00208.svg" style="zoom:50%;" /> |
| 12 | c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /> | 10 | 4.2 | DB00763<br>DB01033 | <img src="smilessvg/DB00763.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01033.svg" style="zoom:50%;" /> |
| 12 | C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /> | 8 | 3.36 | DB00686 | <img src="smilessvg/DB00686.svg" style="zoom:50%;" /> |
| 12 | C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /> | 5 | 2.1 | DB01395 | <img src="smilessvg/DB01395.svg" style="zoom:50%;" /> |
| 13 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 4 | 8.89 | DB00251<br>DB01167<br>DB01263 | <img src="smilessvg/DB00251.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01263.svg" style="zoom:50%;" /> |
| 13 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 3 | 6.67 | DB01188<br>DB01243<br>DB01422 | <img src="smilessvg/DB01188.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01243.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01422.svg" style="zoom:50%;" /> |
| 13 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 2 | 4.44 | DB06697<br>DB00646 | <img src="smilessvg/DB06697.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00646.svg" style="zoom:50%;" /> |
| 14 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 43 | 11.91 | DB01223<br>DB00651<br>DB00277 | <img src="smilessvg/DB01223.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00651.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00277.svg" style="zoom:50%;" /> |
| 14 | c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /> | 28 | 7.76 | DB01223<br>DB00277 | <img src="smilessvg/DB01223.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00277.svg" style="zoom:50%;" /> |
| 14 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 25 | 6.93 | DB08807 | <img src="smilessvg/DB08807.svg" style="zoom:50%;" /> |
| 15 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 20 | 9.9 | DB01030<br>DB00724<br>DB00537 | <img src="smilessvg/DB01030.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00724.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /> |
| 15 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 15 | 7.43 | DB09054<br>DB00441<br>DB00619 | <img src="smilessvg/DB09054.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00441.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /> |
| 15 | C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /> | 11 | 5.45 | DB00441<br>DB00631<br>DB01262 | <img src="smilessvg/DB00441.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00631.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01262.svg" style="zoom:50%;" /> |
| 16 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 985 | 18.12 | DB00557<br>DB00370<br>DB09068 | <img src="smilessvg/DB00557.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00370.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09068.svg" style="zoom:50%;" /> |
| 16 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 810 | 14.9 | DB00425<br>DB08883<br>DB00370 | <img src="smilessvg/DB00425.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08883.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00370.svg" style="zoom:50%;" /> |
| 16 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 750 | 13.8 | DB00956<br>DB00921<br>DB00611 | <img src="smilessvg/DB00956.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00921.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00611.svg" style="zoom:50%;" /> |
| 17 | c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /> | 37 | 44.58 | DB01226<br>DB01199<br>DB00565 | <img src="smilessvg/DB01226.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01199.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /> |
| 17 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 15 | 18.07 | DB00214 | <img src="smilessvg/DB00214.svg" style="zoom:50%;" /> |
| 17 | c1cOc2ccc(cc2)C[C@@H]2NCCc3ccc(cc32)Oc2cccc(c2)CC1<br><img src="motifsvgg/motif_4817.svg" style="zoom:60%;" /> | 5 | 6.02 | DB01199 | <img src="smilessvg/DB01199.svg" style="zoom:50%;" /> |
| 18 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 52 | 63.41 | DB06701<br>DB00422 | <img src="smilessvg/DB06701.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /> |
| 18 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 11 | 13.41 | DB00884 | <img src="smilessvg/DB00884.svg" style="zoom:50%;" /> |
| 18 | c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /> | 6 | 7.32 | DB00927<br>DB00585 | <img src="smilessvg/DB00927.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00585.svg" style="zoom:50%;" /> |
| 19 | C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /> | 76 | 69.72 | DB00353<br>DB01253<br>DB00696 | <img src="smilessvg/DB00353.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00696.svg" style="zoom:50%;" /> |
| 19 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 33 | 30.28 | DB00248<br>DB00320 | <img src="smilessvg/DB00248.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00320.svg" style="zoom:50%;" /> |
| 19 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 15 | 13.76 | DB01200<br>DB01186<br>DB08807 | <img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01186.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /> |
| 20 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 1248 | 20.33 | DB01165<br>DB00537<br>DB01137 | <img src="smilessvg/DB01165.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01137.svg" style="zoom:50%;" /> |
| 20 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 993 | 16.17 | DB00537<br>DB00875<br>DB00363 | <img src="smilessvg/DB00537.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00363.svg" style="zoom:50%;" /> |
| 20 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 683 | 11.12 | DB00502<br>DB04844<br>DB01267 | <img src="smilessvg/DB00502.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04844.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /> |
| 21 | c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /> | 120 | 28.04 | DB01226<br>DB00565<br>DB01199 | <img src="smilessvg/DB01226.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01199.svg" style="zoom:50%;" /> |
| 21 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 34 | 7.94 | DB00908<br>DB00468<br>DB01427 | <img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01427.svg" style="zoom:50%;" /> |
| 21 | C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /> | 32 | 7.48 | DB01627<br>DB01190 | <img src="smilessvg/DB01627.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01190.svg" style="zoom:50%;" /> |
| 22 | c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /> | 47 | 50.0 | DB01226 | <img src="smilessvg/DB01226.svg" style="zoom:50%;" /> |
| 22 | c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /> | 2 | 2.13 | DB08906 | <img src="smilessvg/DB08906.svg" style="zoom:50%;" /> |
| 22 | C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /> | 2 | 2.13 | DB00288 | <img src="smilessvg/DB00288.svg" style="zoom:50%;" /> |
| 23 | c1cNccCC1<br><img src="motifsvgg/motif_7515.svg" style="zoom:60%;" /> | 18 | 32.14 | DB00726<br>DB00458<br>DB01151 | <img src="smilessvg/DB00726.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00458.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01151.svg" style="zoom:50%;" /> |
| 23 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 6 | 10.71 | DB00543 | <img src="smilessvg/DB00543.svg" style="zoom:50%;" /> |
| 23 | C1=NccOcc1<br><img src="motifsvgg/motif_3711.svg" style="zoom:60%;" /> | 6 | 10.71 | DB00543 | <img src="smilessvg/DB00543.svg" style="zoom:50%;" /> |
| 24 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 54 | 30.0 | DB01421<br>DB00919<br>DB01172 | <img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00919.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /> |
| 24 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 20 | 11.11 | DB00884 | <img src="smilessvg/DB00884.svg" style="zoom:50%;" /> |
| 24 | C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /> | 18 | 10.0 | DB01421<br>DB00452 | <img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /> |
| 25 | C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /> | 134 | 18.72 | DB00575<br>DB06694<br>DB06711 | <img src="smilessvg/DB00575.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06694.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06711.svg" style="zoom:50%;" /> |
| 25 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 101 | 14.11 | DB08807<br>DB01136<br>DB01200 | <img src="smilessvg/DB08807.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01200.svg" style="zoom:50%;" /> |
| 25 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 55 | 7.68 | DB00320<br>DB01267<br>DB00734 | <img src="smilessvg/DB00320.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00734.svg" style="zoom:50%;" /> |
| 26 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 7 | 100.0 | DB00758<br>DB06209 | <img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /> |
| 26 | c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /> | 5 | 71.43 | DB00758 | <img src="smilessvg/DB00758.svg" style="zoom:50%;" /> |
| 26 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 1 | 14.29 | DB00295 | <img src="smilessvg/DB00295.svg" style="zoom:50%;" /> |
| 27 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 618 | 66.03 | DB04576<br>DB00817<br>DB00487 | <img src="smilessvg/DB04576.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00817.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00487.svg" style="zoom:50%;" /> |
| 27 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 153 | 16.35 | DB01059<br>DB01208<br>DB01149 | <img src="smilessvg/DB01059.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01208.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01149.svg" style="zoom:50%;" /> |
| 27 | c1ccnnc1<br><img src="motifsvgg/motif_5390.svg" style="zoom:60%;" /> | 87 | 9.29 | DB00827<br>DB00972<br>DB00805 | <img src="smilessvg/DB00827.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00972.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00805.svg" style="zoom:50%;" /> |
| 28 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 1 | 9.09 | DB00619 | <img src="smilessvg/DB00619.svg" style="zoom:50%;" /> |
| 28 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 1 | 9.09 | DB00619 | <img src="smilessvg/DB00619.svg" style="zoom:50%;" /> |
| 28 | C1ccCN1<br><img src="motifsvgg/motif_1707.svg" style="zoom:60%;" /> | 1 | 9.09 | DB01041 | <img src="smilessvg/DB01041.svg" style="zoom:50%;" /> |
| 29 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 96 | 29.63 | DB00214<br>DB00608<br>DB00469 | <img src="smilessvg/DB00214.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /> |
| 29 | C1=CNScc1<br><img src="motifsvgg/motif_2638.svg" style="zoom:60%;" /> | 16 | 4.94 | DB00554<br>DB00469<br>DB06725 | <img src="smilessvg/DB00554.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06725.svg" style="zoom:50%;" /> |
| 29 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 8 | 2.47 | DB00744<br>DB01600 | <img src="smilessvg/DB00744.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01600.svg" style="zoom:50%;" /> |
| 30 | C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /> | 107 | 17.37 | DB01351<br>DB01154<br>DB00849 | <img src="smilessvg/DB01351.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01154.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00849.svg" style="zoom:50%;" /> |
| 30 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 89 | 14.45 | DB00457<br>DB00590<br>DB01162 | <img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /> |
| 30 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 83 | 13.47 | DB00457<br>DB00590<br>DB01162 | <img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /> |
| 31 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 2 | 14.29 | DB09245<br>DB00614 | <img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /> |
| 31 | C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /> | 2 | 14.29 | DB01171<br>DB00805 | <img src="smilessvg/DB01171.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00805.svg" style="zoom:50%;" /> |
| 31 | c1c[s+]ccn1<br><img src="motifsvgg/motif_6844.svg" style="zoom:60%;" /> | 1 | 7.14 | DB09241 | <img src="smilessvg/DB09241.svg" style="zoom:50%;" /> |
| 32 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 277 | 27.29 | DB05271<br>DB00334<br>DB09017 | <img src="smilessvg/DB05271.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09017.svg" style="zoom:50%;" /> |
| 32 | c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /> | 255 | 25.12 | DB00413 | <img src="smilessvg/DB00413.svg" style="zoom:50%;" /> |
| 32 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 104 | 10.25 | DB00334<br>DB01224<br>DB00656 | <img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00656.svg" style="zoom:50%;" /> |
| 33 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 154 | 34.76 | DB08881<br>DB00908<br>DB00468 | <img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /> |
| 33 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 112 | 25.28 | DB08881<br>DB00150<br>DB11699 | <img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00150.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11699.svg" style="zoom:50%;" /> |
| 33 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 97 | 21.9 | DB11730<br>DB00875<br>DB01624 | <img src="smilessvg/DB11730.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01624.svg" style="zoom:50%;" /> |
| 34 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 121 | 38.78 | DB08895<br>DB01280<br>DB08877 | <img src="smilessvg/DB08895.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /> |
| 34 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 113 | 36.22 | DB08895<br>DB08877<br>DB06603 | <img src="smilessvg/DB08895.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06603.svg" style="zoom:50%;" /> |
| 34 | C1CNCCC1<br><img src="motifsvgg/motif_9716.svg" style="zoom:60%;" /> | 101 | 32.37 | DB08895 | <img src="smilessvg/DB08895.svg" style="zoom:50%;" /> |
| 35 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 19 | 27.54 | DB00850<br>DB00875<br>DB09128 | <img src="smilessvg/DB00850.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09128.svg" style="zoom:50%;" /> |
| 35 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 10 | 14.49 | DB09224<br>DB06144<br>DB01267 | <img src="smilessvg/DB09224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /> |
| 35 | c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /> | 6 | 8.7 | DB00477<br>DB01614<br>DB00433 | <img src="smilessvg/DB00477.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00433.svg" style="zoom:50%;" /> |
| 36 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 34 | 36.17 | DB00298<br>DB01238<br>DB00850 | <img src="smilessvg/DB00298.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00850.svg" style="zoom:50%;" /> |
| 36 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 20 | 21.28 | DB06144<br>DB01608<br>DB09286 | <img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09286.svg" style="zoom:50%;" /> |
| 36 | c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /> | 12 | 12.77 | DB00679<br>DB01608<br>DB00477 | <img src="smilessvg/DB00679.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00477.svg" style="zoom:50%;" /> |
| 37 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 322 | 10.42 | DB01392<br>DB08807<br>DB01136 | <img src="smilessvg/DB01392.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01136.svg" style="zoom:50%;" /> |
| 37 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 166 | 5.37 | DB06725<br>DB04951<br>DB00469 | <img src="smilessvg/DB06725.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /> |
| 37 | c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /> | 155 | 5.02 | DB00594<br>DB00384<br>DB00484 | <img src="smilessvg/DB00594.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00484.svg" style="zoom:50%;" /> |
| 38 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 6 | 18.18 | DB06237 | <img src="smilessvg/DB06237.svg" style="zoom:50%;" /> |
| 38 | c1cOCO1<br><img src="motifsvgg/motif_613.svg" style="zoom:60%;" /> | 6 | 18.18 | DB00820 | <img src="smilessvg/DB00820.svg" style="zoom:50%;" /> |
| 38 | c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /> | 6 | 18.18 | DB00820 | <img src="smilessvg/DB00820.svg" style="zoom:50%;" /> |
| 39 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 28 | 18.92 | DB00652<br>DB00844<br>DB00497 | <img src="smilessvg/DB00652.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00497.svg" style="zoom:50%;" /> |
| 39 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 12 | 8.11 | DB00454<br>DB00813<br>DB00967 | <img src="smilessvg/DB00454.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /> |
| 39 | c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /> | 8 | 5.41 | DB00956<br>DB00318<br>DB01466 | <img src="smilessvg/DB00956.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /> |
| 40 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 90 | 30.0 | DB03615<br>DB01421<br>DB01172 | <img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /> |
| 40 | c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /> | 80 | 26.67 | DB01336<br>DB00565<br>DB01226 | <img src="smilessvg/DB01336.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01226.svg" style="zoom:50%;" /> |
| 40 | C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /> | 30 | 10.0 | DB01421<br>DB00452 | <img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /> |
| 41 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 14 | 100.0 | DB00454 | <img src="smilessvg/DB00454.svg" style="zoom:50%;" /> |
| 41 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 2 | 14.29 | DB00614<br>DB09245 | <img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /> |
| 41 | C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /> | 2 | 14.29 | DB00805<br>DB01171 | <img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /> |
| 42 | c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /> | 4 | 66.67 | DB00384<br>DB00594 | <img src="smilessvg/DB00384.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00594.svg" style="zoom:50%;" /> |
| 42 | C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /> | 3 | 50.0 | DB00091 | <img src="smilessvg/DB00091.svg" style="zoom:50%;" /> |
| 42 | C1CCCCOCCNCCC2CCC[C@@H](CCCC\C=C\C1)O2<br><img src="motifsvgg/motif_11033.svg" style="zoom:60%;" /> | 3 | 50.0 | DB00864 | <img src="smilessvg/DB00864.svg" style="zoom:50%;" /> |
| 43 | C1=CNCSC1<br><img src="motifsvgg/motif_10059.svg" style="zoom:60%;" /> | 2 | 18.18 | DB01327 | <img src="smilessvg/DB01327.svg" style="zoom:50%;" /> |
| 43 | c1nncs1<br><img src="motifsvgg/motif_7505.svg" style="zoom:60%;" /> | 2 | 18.18 | DB01327 | <img src="smilessvg/DB01327.svg" style="zoom:50%;" /> |
| 43 | C1CNCN1<br><img src="motifsvgg/motif_5848.svg" style="zoom:60%;" /> | 1 | 9.09 | DB00252 | <img src="smilessvg/DB00252.svg" style="zoom:50%;" /> |
| 44 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 2 | 15.38 | DB09245<br>DB00614 | <img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /> |
| 44 | C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /> | 2 | 15.38 | DB00805<br>DB01171 | <img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /> |
| 44 | c1ccnnc1<br><img src="motifsvgg/motif_5390.svg" style="zoom:60%;" /> | 1 | 7.69 | DB00805 | <img src="smilessvg/DB00805.svg" style="zoom:50%;" /> |
| 45 | C1C[C@@H]2CCC[C@H]1N2<br><img src="motifsvgg/motif_4999.svg" style="zoom:60%;" /> | 33 | 100.0 | DB08824<br>DB00907 | <img src="smilessvg/DB08824.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00907.svg" style="zoom:50%;" /> |
| 45 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 3 | 9.09 | DB01149<br>DB00490<br>DB00543 | <img src="smilessvg/DB01149.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00490.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00543.svg" style="zoom:50%;" /> |
| 45 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 3 | 9.09 | DB00813<br>DB00422<br>DB06701 | <img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06701.svg" style="zoom:50%;" /> |
| 46 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 2 | 7.69 | DB01136<br>DB08807 | <img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /> |
| 46 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 1 | 3.85 | DB09204 | <img src="smilessvg/DB09204.svg" style="zoom:50%;" /> |
| 46 | c1cOCCC1<br><img src="motifsvgg/motif_6858.svg" style="zoom:60%;" /> | 1 | 3.85 | DB04861 | <img src="smilessvg/DB04861.svg" style="zoom:50%;" /> |
| 47 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 4764 | 13.86 | DB00224<br>DB01026<br>DB00619 | <img src="smilessvg/DB00224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01026.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /> |
| 47 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 4071 | 11.85 | DB04868<br>DB00619<br>DB09054 | <img src="smilessvg/DB04868.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09054.svg" style="zoom:50%;" /> |
| 47 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 2991 | 8.7 | DB00951<br>DB00468<br>DB00608 | <img src="smilessvg/DB00951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /> |
| 47 | C1=CC=CS1<br /><img src="motifsvgg/motif_t47.svg" style="zoom:60%;" /> | 1913 | 5.57 | DB00208<br />DB00758<br />DB00744 | <img src="smilessvg/DB00208.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00744.svg" style="zoom:50%;" /> |
| 47 | C12=CC=CC=C1C=CC=N2<br /><img src="motifsvgg/motif_t472.svg" style="zoom:60%;" /> | 1884 | 5.48 | DB01232<br />DB00468<br />DB00608 | <img src="smilessvg/DB01232.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00608.svg" style="zoom:50%;" /> |
| 48 | c1c[n+]ccn1<br><img src="motifsvgg/motif_9727.svg" style="zoom:60%;" /> | 14 | 20.29 | DB09055 | <img src="smilessvg/DB09055.svg" style="zoom:50%;" /> |
| 48 | C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /> | 7 | 10.14 | DB00227 | <img src="smilessvg/DB00227.svg" style="zoom:50%;" /> |
| 48 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 6 | 8.7 | DB08860 | <img src="smilessvg/DB08860.svg" style="zoom:50%;" /> |
| 49 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 7144 | 11.71 | DB09286<br>DB00454<br>DB01002 | <img src="smilessvg/DB09286.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01002.svg" style="zoom:50%;" /> |
| 49 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 7030 | 11.53 | DB00363<br>DB01224<br>DB00298 | <img src="smilessvg/DB00363.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00298.svg" style="zoom:50%;" /> |
| 49 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 3965 | 6.5 | DB00295<br>DB00497<br>DB00844 | <img src="smilessvg/DB00295.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00497.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /> |
| 50 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 2 | 7.69 | DB06203<br>DB08882 | <img src="smilessvg/DB06203.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08882.svg" style="zoom:50%;" /> |
| 50 | C1CCCCN1<br><img src="motifsvgg/motif_599.svg" style="zoom:60%;" /> | 2 | 7.69 | DB00491<br>DB00419 | <img src="smilessvg/DB00491.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00419.svg" style="zoom:50%;" /> |
| 50 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 1 | 3.85 | DB08907 | <img src="smilessvg/DB08907.svg" style="zoom:50%;" /> |
| 51 | c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /> | 7 | 8.43 | DB00774 | <img src="smilessvg/DB00774.svg" style="zoom:50%;" /> |
| 51 | c1cnoc1<br><img src="motifsvgg/motif_3177.svg" style="zoom:60%;" /> | 6 | 7.23 | DB01406 | <img src="smilessvg/DB01406.svg" style="zoom:50%;" /> |
| 51 | S<br><img src="motifsvgg/motif_12299.svg" style="zoom:60%;" /> | 77 | 92.77 | DB01021<br>DB00232<br>DB01324 | <img src="smilessvg/DB01021.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00232.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01324.svg" style="zoom:50%;" /> |
| 52 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 4 | 40.0 | DB11699<br>DB00757 | <img src="smilessvg/DB11699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00757.svg" style="zoom:50%;" /> |
| 52 | C1ccCCN1<br><img src="motifsvgg/motif_3602.svg" style="zoom:60%;" /> | 2 | 20.0 | DB00377 | <img src="smilessvg/DB00377.svg" style="zoom:50%;" /> |
| 52 | C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /> | 2 | 20.0 | DB11699 | <img src="smilessvg/DB11699.svg" style="zoom:50%;" /> |
| 53 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 39 | 11.61 | DB00235<br>DB05266 | <img src="smilessvg/DB00235.svg" style="zoom:50%;" /><br><img src="smilessvg/DB05266.svg" style="zoom:50%;" /> |
| 53 | C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /> | 29 | 8.63 | DB01296 | <img src="smilessvg/DB01296.svg" style="zoom:50%;" /> |
| 53 | C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /> | 27 | 8.04 | DB01240 | <img src="smilessvg/DB01240.svg" style="zoom:50%;" /> |
| 54 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 186 | 14.57 | DB01136<br>DB06791<br>DB08877 | <img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06791.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /> |
| 54 | C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /> | 124 | 9.71 | DB00390<br>DB00511<br>DB01078 | <img src="smilessvg/DB00390.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00511.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /> |
| 54 | C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /> | 124 | 9.71 | DB00390<br>DB01092<br>DB01078 | <img src="smilessvg/DB00390.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01092.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /> |
| 55 | c1cCOC1<br><img src="motifsvgg/motif_5314.svg" style="zoom:60%;" /> | 20 | 16.95 | DB00215<br>DB01175 | <img src="smilessvg/DB00215.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01175.svg" style="zoom:50%;" /> |
| 55 | c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /> | 11 | 9.32 | DB00774 | <img src="smilessvg/DB00774.svg" style="zoom:50%;" /> |
| 55 | c1cOCO1<br><img src="motifsvgg/motif_613.svg" style="zoom:60%;" /> | 11 | 9.32 | DB00715 | <img src="smilessvg/DB00715.svg" style="zoom:50%;" /> |
| 56 | c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /> | 2 | 6.06 | DB00594<br>DB00384 | <img src="smilessvg/DB00594.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /> |
| 56 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 2 | 6.06 | DB11106<br>DB00384 | <img src="smilessvg/DB11106.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /> |
| 56 | C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /> | 2 | 6.06 | DB00700<br>DB01395 | <img src="smilessvg/DB00700.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01395.svg" style="zoom:50%;" /> |
| 57 | C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /> | 105 | 15.81 | DB00091 | <img src="smilessvg/DB00091.svg" style="zoom:50%;" /> |
| 57 | C1CCCCOCCNCCC2CCC[C@@H](CCCC\C=C\C1)O2<br><img src="motifsvgg/motif_11033.svg" style="zoom:60%;" /> | 81 | 12.2 | DB00864<br>DB00337 | <img src="smilessvg/DB00864.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00337.svg" style="zoom:50%;" /> |
| 57 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 44 | 6.63 | DB01172<br>DB00452<br>DB00684 | <img src="smilessvg/DB01172.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00684.svg" style="zoom:50%;" /> |
| 58 | C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /> | 827 | 79.29 | DB01092<br>DB01078<br>DB01396 | <img src="smilessvg/DB01092.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /> |
| 58 | C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /> | 827 | 79.29 | DB00511<br>DB01078<br>DB01396 | <img src="smilessvg/DB00511.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /> |
| 58 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 100 | 9.59 | DB06771<br>DB08911<br>DB00537 | <img src="smilessvg/DB06771.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08911.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /> |
| 59 | C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /> | 3 | 6.98 | DB00572<br>DB00245<br>DB00424 | <img src="smilessvg/DB00572.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00424.svg" style="zoom:50%;" /> |
| 59 | c1cOccC1<br><img src="motifsvgg/motif_908.svg" style="zoom:60%;" /> | 2 | 4.65 | DB00940<br>DB00782 | <img src="smilessvg/DB00940.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00782.svg" style="zoom:50%;" /> |
| 59 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 2 | 4.65 | DB00340<br>DB00967 | <img src="smilessvg/DB00340.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /> |
| 60 | C1=CNC=CC1<br><img src="motifsvgg/motif_7812.svg" style="zoom:60%;" /> | 1137 | 13.54 | DB09236<br>DB00528<br>DB06712 | <img src="smilessvg/DB09236.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00528.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06712.svg" style="zoom:50%;" /> |
| 60 | C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /> | 974 | 11.6 | DB00599<br>DB00474<br>DB01351 | <img src="smilessvg/DB00599.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00474.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01351.svg" style="zoom:50%;" /> |
| 60 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 920 | 10.96 | DB00734<br>DB00346<br>DB08932 | <img src="smilessvg/DB00734.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00346.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08932.svg" style="zoom:50%;" /> |
| 61 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 166 | 33.33 | DB09128<br>DB00831<br>DB01624 | <img src="smilessvg/DB09128.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00831.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01624.svg" style="zoom:50%;" /> |
| 61 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 96 | 19.28 | DB01100<br>DB06144<br>DB04842 | <img src="smilessvg/DB01100.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04842.svg" style="zoom:50%;" /> |
| 61 | c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /> | 53 | 10.64 | DB01614<br>DB00433<br>DB01608 | <img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00433.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /> |
| 62 | C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /> | 5 | 45.45 | DB01200<br>DB01253<br>DB00353 | <img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00353.svg" style="zoom:50%;" /> |
| 62 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 4 | 36.36 | DB00950<br>DB00699<br>DB00248 | <img src="smilessvg/DB00950.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00248.svg" style="zoom:50%;" /> |
| 62 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 2 | 18.18 | DB01200<br>DB01186 | <img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01186.svg" style="zoom:50%;" /> |
| 63 | C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /> | 6 | 17.65 | DB01348<br>DB00492<br>DB00584 | <img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00492.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /> |
| 63 | c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /> | 3 | 8.82 | DB01033 | <img src="smilessvg/DB01033.svg" style="zoom:50%;" /> |
| 63 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 2 | 5.88 | DB00440 | <img src="smilessvg/DB00440.svg" style="zoom:50%;" /> |
| 64 | c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /> | 168 | 20.92 | DB00150<br>DB11699<br>DB00757 | <img src="smilessvg/DB00150.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00757.svg" style="zoom:50%;" /> |
| 64 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 152 | 18.93 | DB09042<br>DB00601<br>DB09245 | <img src="smilessvg/DB09042.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00601.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /> |
| 64 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 108 | 13.45 | DB09209<br>DB00854<br>DB00652 | <img src="smilessvg/DB09209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00854.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00652.svg" style="zoom:50%;" /> |
| 65 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 14 | 43.75 | DB01192<br>DB01466<br>DB00295 | <img src="smilessvg/DB01192.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00295.svg" style="zoom:50%;" /> |
| 65 | c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /> | 4 | 12.5 | DB01466<br>DB00318<br>DB00956 | <img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00956.svg" style="zoom:50%;" /> |
| 65 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 4 | 12.5 | DB01081<br>DB00813<br>DB00454 | <img src="smilessvg/DB01081.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /> |
| 66 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 93 | 72.66 | DB12364<br>DB09075<br>DB00608 | <img src="smilessvg/DB12364.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /> |
| 66 | c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /> | 5 | 3.91 | DB06228<br>DB00744<br>DB01600 | <img src="smilessvg/DB06228.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00744.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01600.svg" style="zoom:50%;" /> |
| 66 | C1=CNScc1<br><img src="motifsvgg/motif_2638.svg" style="zoom:60%;" /> | 4 | 3.12 | DB00814<br>DB00554<br>DB06725 | <img src="smilessvg/DB00814.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00554.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06725.svg" style="zoom:50%;" /> |
| 67 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 184 | 16.85 | DB00467<br>DB01137<br>DB01044 | <img src="smilessvg/DB00467.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01137.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01044.svg" style="zoom:50%;" /> |
| 67 | c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /> | 108 | 9.89 | DB01069<br>DB01614<br>DB01246 | <img src="smilessvg/DB01069.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01246.svg" style="zoom:50%;" /> |
| 67 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 101 | 9.25 | DB01059<br>DB01208<br>DB00850 | <img src="smilessvg/DB01059.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01208.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00850.svg" style="zoom:50%;" /> |
| 68 | C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /> | 137 | 49.28 | DB01395<br>DB00700 | <img src="smilessvg/DB01395.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00700.svg" style="zoom:50%;" /> |
| 68 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 52 | 18.71 | DB00440<br>DB01349<br>DB00384 | <img src="smilessvg/DB00440.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01349.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /> |
| 68 | C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /> | 45 | 16.19 | DB01348<br>DB09477<br>DB00584 | <img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09477.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /> |
| 69 | c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /> | 137 | 48.93 | DB00611<br>DB01466<br>DB00652 | <img src="smilessvg/DB00611.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00652.svg" style="zoom:50%;" /> |
| 69 | c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /> | 40 | 14.29 | DB01466<br>DB01551<br>DB00318 | <img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01551.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /> |
| 69 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 38 | 13.57 | DB06738<br>DB01081<br>DB00454 | <img src="smilessvg/DB06738.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01081.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /> |
| 70 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 453 | 5.82 | DB06698<br>DB01132<br>DB00608 | <img src="smilessvg/DB06698.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01132.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /> |
| 70 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 433 | 5.56 | DB06148<br>DB00334<br>DB08815 | <img src="smilessvg/DB06148.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08815.svg" style="zoom:50%;" /> |
| 70 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 405 | 5.2 | DB00358<br>DB09097<br>DB00843 | <img src="smilessvg/DB00358.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09097.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00843.svg" style="zoom:50%;" /> |
| 71 | C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /> | 172 | 25.56 | DB00696<br>DB01253<br>DB00353 | <img src="smilessvg/DB00696.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00353.svg" style="zoom:50%;" /> |
| 71 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 127 | 18.87 | DB09245<br>DB00614<br>DB00601 | <img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00601.svg" style="zoom:50%;" /> |
| 71 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 119 | 17.68 | DB00248<br>DB00320<br>DB00422 | <img src="smilessvg/DB00248.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00320.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /> |
| 72 | C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /> | 473 | 25.92 | DB00684<br>DB03615<br>DB00452 | <img src="smilessvg/DB00684.svg" style="zoom:50%;" /><br><img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /> |
| 72 | C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /> | 154 | 8.44 | DB00452<br>DB01421 | <img src="smilessvg/DB00452.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /> |
| 72 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 117 | 6.41 | DB00469<br>DB04951<br>DB04552 | <img src="smilessvg/DB00469.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04552.svg" style="zoom:50%;" /> |
| 73 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 3434 | 14.44 | DB09073<br>DB00243<br>DB01149 | <img src="smilessvg/DB09073.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00243.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01149.svg" style="zoom:50%;" /> |
| 73 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 3081 | 12.96 | DB09048<br>DB08881<br>DB01220 | <img src="smilessvg/DB09048.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01220.svg" style="zoom:50%;" /> |
| 73 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 2828 | 11.89 | DB09330<br>DB01254<br>DB08864 | <img src="smilessvg/DB09330.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08864.svg" style="zoom:50%;" /> |
| 73 | C1(C=CC=C2)=C2OCO1<br /><img src="motifsvgg/motif_t72.svg" style="zoom:60%;" /> | 536 | 2.25 | DB09118<br />DB00715 <br />DB00820 | <img src="smilessvg/DB09118.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00715.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00820.svg" style="zoom:50%;" /> |
| 74 | C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /> | 47 | 11.14 | DB00621 | <img src="smilessvg/DB00621.svg" style="zoom:50%;" /> |
| 74 | c1cn[nH]c1<br><img src="motifsvgg/motif_980.svg" style="zoom:60%;" /> | 47 | 11.14 | DB06718 | <img src="smilessvg/DB06718.svg" style="zoom:50%;" /> |
| 74 | c1cnoc1<br><img src="motifsvgg/motif_3177.svg" style="zoom:60%;" /> | 47 | 11.14 | DB01406 | <img src="smilessvg/DB01406.svg" style="zoom:50%;" /> |
| 75 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 1472 | 15.48 | DB09280<br>DB08881<br>DB00238 | <img src="smilessvg/DB09280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00238.svg" style="zoom:50%;" /> |
| 75 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 989 | 10.4 | DB08864<br>DB06414<br>DB09330 | <img src="smilessvg/DB08864.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06414.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09330.svg" style="zoom:50%;" /> |
| 75 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 667 | 7.02 | DB01238<br>DB01167<br>DB00224 | <img src="smilessvg/DB01238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00224.svg" style="zoom:50%;" /> |
| 76 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 81 | 14.52 | DB01176<br>DB04841<br>DB00354 | <img src="smilessvg/DB01176.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04841.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00354.svg" style="zoom:50%;" /> |
| 76 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 79 | 14.16 | DB00719<br>DB00455<br>DB00920 | <img src="smilessvg/DB00719.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00455.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00920.svg" style="zoom:50%;" /> |
| 76 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 57 | 10.22 | DB00792<br>DB00719<br>DB00967 | <img src="smilessvg/DB00792.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00719.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /> |
| 77 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 68 | 12.64 | DB04946<br>DB00615<br>DB00455 | <img src="smilessvg/DB04946.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00455.svg" style="zoom:50%;" /> |
| 77 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 52 | 9.67 | DB01026<br>DB01263<br>DB01167 | <img src="smilessvg/DB01026.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01263.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /> |
| 77 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 31 | 5.76 | DB00908<br>DB00967<br>DB09280 | <img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09280.svg" style="zoom:50%;" /> |
| 78 | C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /> | 1 | 3.57 | DB00288 | <img src="smilessvg/DB00288.svg" style="zoom:50%;" /> |
| 78 | S<br><img src="motifsvgg/motif_12299.svg" style="zoom:60%;" /> | 28 | 100.0 | DB09063 | <img src="smilessvg/DB09063.svg" style="zoom:50%;" /> |
| 78 | O=S<br><img src="motifsvgg/motif_5204.svg" style="zoom:60%;" /> | 28 | 100.0 | DB09063 | <img src="smilessvg/DB09063.svg" style="zoom:50%;" /> |
| 79 | C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /> | 8 | 29.63 | DB00614<br>DB09245 | <img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /> |
| 79 | C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /> | 7 | 25.93 | DB00805<br>DB01171 | <img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /> |
| 79 | C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /> | 5 | 18.52 | DB00813 | <img src="smilessvg/DB00813.svg" style="zoom:50%;" /> |
| 80 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 9 | 14.06 | DB00951<br>DB00908<br>DB00613 | <img src="smilessvg/DB00951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00613.svg" style="zoom:50%;" /> |
| 80 | C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /> | 9 | 14.06 | DB00091 | <img src="smilessvg/DB00091.svg" style="zoom:50%;" /> |
| 80 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 8 | 12.5 | DB00619<br>DB01254<br>DB00943 | <img src="smilessvg/DB00619.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00943.svg" style="zoom:50%;" /> |
| 81 | C1ccCN1<br><img src="motifsvgg/motif_1707.svg" style="zoom:60%;" /> | 32 | 29.63 | DB01041<br>DB00480 | <img src="smilessvg/DB01041.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00480.svg" style="zoom:50%;" /> |
| 81 | C1CCCCN1<br><img src="motifsvgg/motif_599.svg" style="zoom:60%;" /> | 32 | 29.63 | DB01041<br>DB00480 | <img src="smilessvg/DB01041.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00480.svg" style="zoom:50%;" /> |
| 81 | C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /> | 2 | 1.85 | DB01395 | <img src="smilessvg/DB01395.svg" style="zoom:50%;" /> |
| 82 | C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /> | 111 | 31.27 | DB01078<br>DB01396<br>DB00511 | <img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00511.svg" style="zoom:50%;" /> |
| 82 | C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /> | 111 | 31.27 | DB01078<br>DB01396<br>DB01092 | <img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01092.svg" style="zoom:50%;" /> |
| 82 | c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /> | 59 | 16.62 | DB01071<br>DB00679<br>DB01608 | <img src="smilessvg/DB01071.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00679.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /> |
| 83 | c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /> | 74 | 6.15 | DB00774 | <img src="smilessvg/DB00774.svg" style="zoom:50%;" /> |
| 83 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 74 | 6.15 | DB00214 | <img src="smilessvg/DB00214.svg" style="zoom:50%;" /> |
| 83 | C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /> | 32 | 2.66 | DB00288<br>DB00273 | <img src="smilessvg/DB00288.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00273.svg" style="zoom:50%;" /> |
| 84 | c1cNccCC1<br><img src="motifsvgg/motif_7515.svg" style="zoom:60%;" /> | 20 | 30.77 | DB01242<br>DB00726<br>DB00458 | <img src="smilessvg/DB01242.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00726.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00458.svg" style="zoom:50%;" /> |
| 84 | C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /> | 10 | 15.38 | DB00370<br>DB00543 | <img src="smilessvg/DB00370.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00543.svg" style="zoom:50%;" /> |
| 84 | c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /> | 5 | 7.69 | DB00370 | <img src="smilessvg/DB00370.svg" style="zoom:50%;" /> |
| 85 | c1cOCCC1<br><img src="motifsvgg/motif_6858.svg" style="zoom:60%;" /> | 149 | 40.05 | DB00486<br>DB00470<br>DB04861 | <img src="smilessvg/DB00486.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00470.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04861.svg" style="zoom:50%;" /> |
| 85 | C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /> | 28 | 7.53 | DB06711<br>DB00751<br>DB00964 | <img src="smilessvg/DB06711.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00751.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00964.svg" style="zoom:50%;" /> |
| 85 | C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /> | 6 | 1.61 | DB00245<br>DB00424<br>DB00572 | <img src="smilessvg/DB00245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00424.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00572.svg" style="zoom:50%;" /> |
| 86 | c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /> | 27 | 100.0 | DB00437 | <img src="smilessvg/DB00437.svg" style="zoom:50%;" /> |
| 86 | c1cn[nH]c1<br><img src="motifsvgg/motif_980.svg" style="zoom:60%;" /> | 27 | 100.0 | DB00437 | <img src="smilessvg/DB00437.svg" style="zoom:50%;" /> |
| 86 | C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /> | 6 | 22.22 | DB00722<br>DB01348<br>DB00584 | <img src="smilessvg/DB00722.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /> |

```Note: A motif is considered present in a drug if it appears among the top-ranked substructures identified by TRACE for that molecule.```