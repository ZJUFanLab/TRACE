![](https://img.shields.io/badge/version-1.0.0-blue)

# Transformer-based graphs for drug-drug interaction with chemical knowledge embedding

This repository is the official implementation of TRACE.

## Brief introduction

We propose TRACE, a new drug-drug interaction (DDI) prediction method, termed **TR**ansformer-based Graph Representation Le**A**rning with **C**hemical **E**mbedding.

### Model

During the construction of molecular graphs, we embed the elemental information from ElementKG into the original drug molecular graphs, resulting in KG-enhanced molecular graphs that combine both structural features and chemical domain knowledge.

After obtaining the KG-enhanced molecular graphs, we input them into the Graph Transformer module to further extract high-level representations of drug molecules. The Graph Transformer leverages self-attention mechanisms to dynamically aggregate local and global information within the graph, effectively capturing complex structural and chemical patterns critical for DDI prediction. The representations of two drug molecules are then concatenated and fed into a downstream multilayer perceptron (MLP), which is trained to predict DDIs.

![overview.png](overview_v2.png)

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

<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <th>Motif (SMILES + Structure)</th>
    <th>Representative Drug ID(s)</th>
    <th>Drug (Structure)</th>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 4【Metabolism】: The metabolism of #Drug2 can be increased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /></td>
    <td align="left">DB00794<br>DB01174<br>DB00312</td>
    <td align="center"><img src="smilessvg/DB00794.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01174.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00312.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00238<br>DB09280<br>DB01220</td>
    <td align="center"><img src="smilessvg/DB00238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01220.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00615<br>DB01074<br>DB11614</td>
    <td align="center"><img src="smilessvg/DB00615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01074.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11614.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 47【Metabolism】: The metabolism of #Drug2 can be decreased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00224<br>DB01026<br>DB00619</td>
    <td align="center"><img src="smilessvg/DB00224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01026.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB04868<br>DB00619<br>DB09054</td>
    <td align="center"><img src="smilessvg/DB04868.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09054.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00951<br>DB00468<br>DB00608</td>
    <td align="center"><img src="smilessvg/DB00951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1=CC=CS1<br /><img src="motifsvgg/motif_t47.svg" style="zoom:60%;" /></td>
    <td align="left">DB00208<br />DB00758<br />DB00744</td>
    <td align="center"><img src="smilessvg/DB00208.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00744.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C12=CC=CC=C1C=CC=N2<br /><img src="motifsvgg/motif_t472.svg" style="zoom:60%;" /></td>
    <td align="left">DB01232<br />DB00468<br />DB00608</td>
    <td align="center"><img src="smilessvg/DB01232.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00608.svg" style="zoom:50%;" /></td>
  </tr>
    <tr>
    <td colspan="3" align="left"><b>DDI Type 67【Metabolism】: #Drug1 can cause a decrease in the absorption of #Drug2 resulting in a reduced serum concentration and potentially a decrease in efficacy.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00467<br>DB01137<br>DB01044</td>
    <td align="center"><img src="smilessvg/DB00467.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01137.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01044.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /></td>
    <td align="left">DB01069<br>DB01614<br>DB01246</td>
    <td align="center"><img src="smilessvg/DB01069.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01246.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01059<br>DB01208<br>DB00850</td>
    <td align="center"><img src="smilessvg/DB01059.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01208.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00850.svg" style="zoom:50%;" /></td>
  </tr>
   <tr>
    <td colspan="3" align="left"><b>DDI Type 77【Metabolism】: The serum concentration of the active metabolites of #Drug2 can be increased when #Drug2 is used in combination with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB04946<br>DB00615<br>DB00455</td>
    <td align="center"><img src="smilessvg/DB04946.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00455.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01026<br>DB01263<br>DB01167</td>
    <td align="center"><img src="smilessvg/DB01026.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01263.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00908<br>DB00967<br>DB09280</td>
    <td align="center"><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09280.svg" style="zoom:50%;" /></td>
  </tr>
   <tr>
    <td colspan="3" align="left"><b>DDI Type 13【Absorption】: The absorption of #Drug2 can be decreased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00251<br>DB01167<br>DB01263</td>
    <td align="center"><img src="smilessvg/DB00251.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01263.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB01188<br>DB01243<br>DB01422</td>
    <td align="center"><img src="smilessvg/DB01188.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01243.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01422.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB06697<br>DB00646</td>
    <td align="center"><img src="smilessvg/DB06697.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00646.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 18【Absorption】: #Drug1 can cause an increase in the absorption of #Drug2 resulting in an increased serum concentration and potentially a worsening of adverse effects.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB06701<br>DB00422</td>
    <td align="center"><img src="smilessvg/DB06701.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00884</td>
    <td align="center"><img src="smilessvg/DB00884.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /></td>
    <td align="left">DB00927<br>DB00585</td>
    <td align="center"><img src="smilessvg/DB00927.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00585.svg" style="zoom:50%;" /></td>
  </tr>
   <tr>
    <td colspan="3" align="left"><b>DDI Type 73【Distribution】: The serum concentration of #Drug2 can be increased when it is combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB09073<br>DB00243<br>DB01149</td>
    <td align="center"><img src="smilessvg/DB09073.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00243.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01149.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB09048<br>DB08881<br>DB01220</td>
    <td align="center"><img src="smilessvg/DB09048.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01220.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB09330<br>DB01254<br>DB08864</td>
    <td align="center"><img src="smilessvg/DB09330.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08864.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1(C=CC=C2)=C2OCO1<br /><img src="motifsvgg/motif_t72.svg" style="zoom:60%;" /></td>
    <td align="left">DB09118<br />DB00715 <br />DB00820</td>
    <td align="center"><img src="smilessvg/DB09118.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00715.svg" style="zoom:50%;" /><br /><img src="smilessvg/DB00820.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 75【Distribution】: The serum concentration of #Drug2 can be decreased when it is combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB09280<br>DB08881<br>DB00238</td>
    <td align="center"><img src="smilessvg/DB09280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00238.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB08864<br>DB06414<br>DB09330</td>
    <td align="center"><img src="smilessvg/DB08864.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06414.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09330.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01238<br>DB01167<br>DB00224</td>
    <td align="center"><img src="smilessvg/DB01238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01167.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00224.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 65【Excretion】: #Drug1 may increase the excretion rate of #Drug2 which could result in a lower serum level and potentially a reduction in efficacy.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB01192<br>DB01466<br>DB00295</td>
    <td align="center"><img src="smilessvg/DB01192.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00295.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /></td>
    <td align="left">DB01466<br>DB00318<br>DB00956</td>
    <td align="center"><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00956.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB01081<br>DB00813<br>DB00454</td>
    <td align="center"><img src="smilessvg/DB01081.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 72【Excretion】: #Drug1 may decrease the excretion rate of #Drug2 which could result in a higher serum level.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB00684<br>DB03615<br>DB00452</td>
    <td align="center"><img src="smilessvg/DB00684.svg" style="zoom:50%;" /><br><img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /></td>
    <td align="left">DB00452<br>DB01421</td>
    <td align="center"><img src="smilessvg/DB00452.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00469<br>DB04951<br>DB04552</td>
    <td align="center"><img src="smilessvg/DB00469.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04552.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 1: #Drug1 may increase the photosensitizing activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3<br><img src="motifsvgg/motif_10271.svg" style="zoom:60%;" /></td>
    <td align="left">DB00460</td>
    <td align="center"><img src="smilessvg/DB00460.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /></td>
    <td align="left">DB04571<br>DB00553</td>
    <td align="center"><img src="smilessvg/DB04571.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00553.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB01600</td>
    <td align="center"><img src="smilessvg/DB01600.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 2: #Drug1 may increase the anticholinergic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB01409<br>DB06153</td>
    <td align="center"><img src="smilessvg/DB01409.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06153.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB06148<br>DB00670</td>
    <td align="center"><img src="smilessvg/DB06148.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00670.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1C[N+]2CCC1CC2<br><img src="motifsvgg/motif_4789.svg" style="zoom:60%;" /></td>
    <td align="left">DB09076</td>
    <td align="center"><img src="smilessvg/DB09076.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 3: The bioavailability of #Drug2 can be decreased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /></td>
    <td align="left">DB08906</td>
    <td align="center"><img src="smilessvg/DB08906.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /></td>
    <td align="left">DB00288</td>
    <td align="center"><img src="smilessvg/DB00288.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCC1<br><img src="motifsvgg/motif_4874.svg" style="zoom:60%;" /></td>
    <td align="left">DB04574<br>DB00588<br>DB08970</td>
    <td align="center"><img src="smilessvg/DB04574.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00588.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08970.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 5: #Drug1 may decrease the vasoconstricting activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00346<br>DB01162<br>DB00457</td>
    <td align="center"><img src="smilessvg/DB00346.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00457.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01162<br>DB00457<br>DB00590</td>
    <td align="center"><img src="smilessvg/DB01162.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCCO1<br><img src="motifsvgg/motif_3579.svg" style="zoom:60%;" /></td>
    <td align="left">DB00590</td>
    <td align="center"><img src="smilessvg/DB00590.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 6: #Drug1 may increase the anticoagulant activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB09075<br>DB05266<br>DB00235</td>
    <td align="center"><img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB05266.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00235.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB06228<br>DB06209<br>DB00758</td>
    <td align="center"><img src="smilessvg/DB06228.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00758.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /></td>
    <td align="left">DB09075<br>DB01254<br>DB00814</td>
    <td align="center"><img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00814.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 7: #Drug1 may increase the ototoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB03615<br>DB01172<br>DB01421</td>
    <td align="center"><img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /></td>
    <td align="left">DB01421<br>DB00452</td>
    <td align="center"><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /></td>
    <td align="left">DB00955</td>
    <td align="center"><img src="smilessvg/DB00955.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 8: The therapeutic efficacy of #Drug2 can be increased when used in combination with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00295<br>DB01466<br>DB00844</td>
    <td align="center"><img src="smilessvg/DB00295.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB01100<br>DB00297<br>DB01501</td>
    <td align="center"><img src="smilessvg/DB01100.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00297.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01501.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00363<br>DB04908<br>DB00875</td>
    <td align="center"><img src="smilessvg/DB00363.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 9: #Drug1 may increase the hypoglycemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB01132<br>DB00468<br>DB00779</td>
    <td align="center"><img src="smilessvg/DB01132.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00779.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00359<br>DB06203<br>DB08882</td>
    <td align="center"><img src="smilessvg/DB00359.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06203.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08882.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01268<br>DB01200<br>DB06791</td>
    <td align="center"><img src="smilessvg/DB01268.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06791.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 10: #Drug1 may increase the antihypertensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /></td>
    <td align="left">DB00484<br>DB00575<br>DB06694</td>
    <td align="center"><img src="smilessvg/DB00484.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00575.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06694.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB06237<br>DB06403<br>DB09242</td>
    <td align="center"><img src="smilessvg/DB06237.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06403.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09242.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB00820<br>DB00206<br>DB01089</td>
    <td align="center"><img src="smilessvg/DB00820.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00206.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01089.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 11: The serum concentration of the active metabolites of #Drug2 can be reduced when #Drug2 is used in combination with #Drug1 resulting in a loss in efficacy.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB00758<br>DB06209<br>DB00208</td>
    <td align="center"><img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00208.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNPOC1<br><img src="motifsvgg/motif_10011.svg" style="zoom:60%;" /></td>
    <td align="left">DB01181</td>
    <td align="center"><img src="smilessvg/DB01181.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /></td>
    <td align="left">DB00758<br>DB00208</td>
    <td align="center"><img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00208.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 12: #Drug1 may decrease the anticoagulant activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /></td>
    <td align="left">DB00763<br>DB01033</td>
    <td align="center"><img src="smilessvg/DB00763.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01033.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /></td>
    <td align="left">DB00686</td>
    <td align="center"><img src="smilessvg/DB00686.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /></td>
    <td align="left">DB01395</td>
    <td align="center"><img src="smilessvg/DB01395.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 14: #Drug1 may decrease the bronchodilatory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB01223<br>DB00651<br>DB00277</td>
    <td align="center"><img src="smilessvg/DB01223.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00651.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00277.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /></td>
    <td align="left">DB01223<br>DB00277</td>
    <td align="center"><img src="smilessvg/DB01223.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00277.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB08807</td>
    <td align="center"><img src="smilessvg/DB08807.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 15: #Drug1 may increase the cardiotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB01030<br>DB00724<br>DB00537</td>
    <td align="center"><img src="smilessvg/DB01030.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00724.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB09054<br>DB00441<br>DB00619</td>
    <td align="center"><img src="smilessvg/DB09054.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00441.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00619.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /></td>
    <td align="left">DB00441<br>DB00631<br>DB01262</td>
    <td align="center"><img src="smilessvg/DB00441.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00631.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01262.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 16: #Drug1 may increase the central nervous system depressant (CNS depressant) activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00557<br>DB00370<br>DB09068</td>
    <td align="center"><img src="smilessvg/DB00557.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00370.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09068.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00425<br>DB08883<br>DB00370</td>
    <td align="center"><img src="smilessvg/DB00425.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08883.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00370.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00956<br>DB00921<br>DB00611</td>
    <td align="center"><img src="smilessvg/DB00956.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00921.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00611.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 17: #Drug1 may decrease the neuromuscular blocking activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /></td>
    <td align="left">DB01226<br>DB01199<br>DB00565</td>
    <td align="center"><img src="smilessvg/DB01226.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01199.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00214</td>
    <td align="center"><img src="smilessvg/DB00214.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOc2ccc(cc2)C[C@@H]2NCCc3ccc(cc32)Oc2cccc(c2)CC1<br><img src="motifsvgg/motif_4817.svg" style="zoom:60%;" /></td>
    <td align="left">DB01199</td>
    <td align="center"><img src="smilessvg/DB01199.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 19: #Drug1 may increase the vasoconstricting activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /></td>
    <td align="left">DB00353<br>DB01253<br>DB00696</td>
    <td align="center"><img src="smilessvg/DB00353.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00696.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00248<br>DB00320</td>
    <td align="center"><img src="smilessvg/DB00248.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00320.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01200<br>DB01186<br>DB08807</td>
    <td align="center"><img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01186.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 20: #Drug1 may increase the QTc-prolonging activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB01165<br>DB00537<br>DB01137</td>
    <td align="center"><img src="smilessvg/DB01165.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01137.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00537<br>DB00875<br>DB00363</td>
    <td align="center"><img src="smilessvg/DB00537.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00363.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00502<br>DB04844<br>DB01267</td>
    <td align="center"><img src="smilessvg/DB00502.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04844.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 21: #Drug1 may increase the neuromuscular blocking activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /></td>
    <td align="left">DB01226<br>DB00565<br>DB01199</td>
    <td align="center"><img src="smilessvg/DB01226.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01199.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00908<br>DB00468<br>DB01427</td>
    <td align="center"><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01427.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /></td>
    <td align="left">DB01627<br>DB01190</td>
    <td align="center"><img src="smilessvg/DB01627.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01190.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 22: #Drug1 may increase the adverse neuromuscular activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /></td>
    <td align="left">DB01226</td>
    <td align="center"><img src="smilessvg/DB01226.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccoc1<br><img src="motifsvgg/motif_2106.svg" style="zoom:60%;" /></td>
    <td align="left">DB08906</td>
    <td align="center"><img src="smilessvg/DB08906.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /></td>
    <td align="left">DB00288</td>
    <td align="center"><img src="smilessvg/DB00288.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 23: #Drug1 may increase the stimulatory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cNccCC1<br><img src="motifsvgg/motif_7515.svg" style="zoom:60%;" /></td>
    <td align="left">DB00726<br>DB00458<br>DB01151</td>
    <td align="center"><img src="smilessvg/DB00726.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00458.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01151.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00543</td>
    <td align="center"><img src="smilessvg/DB00543.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1=NccOcc1<br><img src="motifsvgg/motif_3711.svg" style="zoom:60%;" /></td>
    <td align="left">DB00543</td>
    <td align="center"><img src="smilessvg/DB00543.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 24: #Drug1 may increase the hypocalcemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB01421<br>DB00919<br>DB01172</td>
    <td align="center"><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00919.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00884</td>
    <td align="center"><img src="smilessvg/DB00884.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /></td>
    <td align="left">DB01421<br>DB00452</td>
    <td align="center"><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 25: #Drug1 may increase the atrioventricular blocking (AV block) activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /></td>
    <td align="left">DB00575<br>DB06694<br>DB06711</td>
    <td align="center"><img src="smilessvg/DB00575.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06694.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06711.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB08807<br>DB01136<br>DB01200</td>
    <td align="center"><img src="smilessvg/DB08807.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01200.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00320<br>DB01267<br>DB00734</td>
    <td align="center"><img src="smilessvg/DB00320.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00734.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 26: #Drug1 may decrease the antiplatelet activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB00758<br>DB06209</td>
    <td align="center"><img src="smilessvg/DB00758.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06209.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /></td>
    <td align="left">DB00758</td>
    <td align="center"><img src="smilessvg/DB00758.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00295</td>
    <td align="center"><img src="smilessvg/DB00295.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 27: #Drug1 may increase the neuroexcitatory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB04576<br>DB00817<br>DB00487</td>
    <td align="center"><img src="smilessvg/DB04576.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00817.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00487.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01059<br>DB01208<br>DB01149</td>
    <td align="center"><img src="smilessvg/DB01059.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01208.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01149.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccnnc1<br><img src="motifsvgg/motif_5390.svg" style="zoom:60%;" /></td>
    <td align="left">DB00827<br>DB00972<br>DB00805</td>
    <td align="center"><img src="smilessvg/DB00827.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00972.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00805.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 28: #Drug1 may increase the dermatologic adverse activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00619</td>
    <td align="center"><img src="smilessvg/DB00619.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00619</td>
    <td align="center"><img src="smilessvg/DB00619.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1ccCN1<br><img src="motifsvgg/motif_1707.svg" style="zoom:60%;" /></td>
    <td align="left">DB01041</td>
    <td align="center"><img src="smilessvg/DB01041.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 29: #Drug1 may decrease the diuretic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00214<br>DB00608<br>DB00469</td>
    <td align="center"><img src="smilessvg/DB00214.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1=CNScc1<br><img src="motifsvgg/motif_2638.svg" style="zoom:60%;" /></td>
    <td align="left">DB00554<br>DB00469<br>DB06725</td>
    <td align="center"><img src="smilessvg/DB00554.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06725.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB00744<br>DB01600</td>
    <td align="center"><img src="smilessvg/DB00744.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01600.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 30: #Drug1 may increase the orthostatic hypotensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /></td>
    <td align="left">DB01351<br>DB01154<br>DB00849</td>
    <td align="center"><img src="smilessvg/DB01351.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01154.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00849.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00457<br>DB00590<br>DB01162</td>
    <td align="center"><img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00457<br>DB00590<br>DB01162</td>
    <td align="center"><img src="smilessvg/DB00457.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00590.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01162.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 31: The risk or severity of hypertension can be increased when #Drug2 is combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB09245<br>DB00614</td>
    <td align="center"><img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /></td>
    <td align="left">DB01171<br>DB00805</td>
    <td align="center"><img src="smilessvg/DB01171.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00805.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1c[s+]ccn1<br><img src="motifsvgg/motif_6844.svg" style="zoom:60%;" /></td>
    <td align="left">DB09241</td>
    <td align="center"><img src="smilessvg/DB09241.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 32: #Drug1 may increase the sedative activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB05271<br>DB00334<br>DB09017</td>
    <td align="center"><img src="smilessvg/DB05271.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09017.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cscn1<br><img src="motifsvgg/motif_10862.svg" style="zoom:60%;" /></td>
    <td align="left">DB00413</td>
    <td align="center"><img src="smilessvg/DB00413.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00334<br>DB01224<br>DB00656</td>
    <td align="center"><img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00656.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 33: The risk or severity of QTc prolongation can be increased when #Drug1 is combined with #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB08881<br>DB00908<br>DB00468</td>
    <td align="center"><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00468.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB08881<br>DB00150<br>DB11699</td>
    <td align="center"><img src="smilessvg/DB08881.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00150.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11699.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB11730<br>DB00875<br>DB01624</td>
    <td align="center"><img src="smilessvg/DB11730.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01624.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 34: #Drug1 may increase the immunosuppressive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB08895<br>DB01280<br>DB08877</td>
    <td align="center"><img src="smilessvg/DB08895.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01280.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB08895<br>DB08877<br>DB06603</td>
    <td align="center"><img src="smilessvg/DB08895.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06603.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCC1<br><img src="motifsvgg/motif_9716.svg" style="zoom:60%;" /></td>
    <td align="left">DB08895</td>
    <td align="center"><img src="smilessvg/DB08895.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 35: #Drug1 may increase the neurotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00850<br>DB00875<br>DB09128</td>
    <td align="center"><img src="smilessvg/DB00850.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00875.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09128.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB09224<br>DB06144<br>DB01267</td>
    <td align="center"><img src="smilessvg/DB09224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01267.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /></td>
    <td align="left">DB00477<br>DB01614<br>DB00433</td>
    <td align="center"><img src="smilessvg/DB00477.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00433.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 36: #Drug1 may increase the antipsychotic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00298<br>DB01238<br>DB00850</td>
    <td align="center"><img src="smilessvg/DB00298.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01238.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00850.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB06144<br>DB01608<br>DB09286</td>
    <td align="center"><img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09286.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /></td>
    <td align="left">DB00679<br>DB01608<br>DB00477</td>
    <td align="center"><img src="smilessvg/DB00679.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00477.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 37: #Drug1 may decrease the antihypertensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01392<br>DB08807<br>DB01136</td>
    <td align="center"><img src="smilessvg/DB01392.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01136.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB06725<br>DB04951<br>DB00469</td>
    <td align="center"><img src="smilessvg/DB06725.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00469.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /></td>
    <td align="left">DB00594<br>DB00384<br>DB00484</td>
    <td align="center"><img src="smilessvg/DB00594.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00484.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 38: #Drug1 may increase the vasodilatory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB06237</td>
    <td align="center"><img src="smilessvg/DB06237.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCO1<br><img src="motifsvgg/motif_613.svg" style="zoom:60%;" /></td>
    <td align="left">DB00820</td>
    <td align="center"><img src="smilessvg/DB00820.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cCNCC1<br><img src="motifsvgg/motif_11990.svg" style="zoom:60%;" /></td>
    <td align="left">DB00820</td>
    <td align="center"><img src="smilessvg/DB00820.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 39: #Drug1 may increase the constipating activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00652<br>DB00844<br>DB00497</td>
    <td align="center"><img src="smilessvg/DB00652.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00497.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00454<br>DB00813<br>DB00967</td>
    <td align="center"><img src="smilessvg/DB00454.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /></td>
    <td align="left">DB00956<br>DB00318<br>DB01466</td>
    <td align="center"><img src="smilessvg/DB00956.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 40: #Drug1 may increase the respiratory depressant activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB03615<br>DB01421<br>DB01172</td>
    <td align="center"><img src="smilessvg/DB03615.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01172.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cC[N+]CC1<br><img src="motifsvgg/motif_8054.svg" style="zoom:60%;" /></td>
    <td align="left">DB01336<br>DB00565<br>DB01226</td>
    <td align="center"><img src="smilessvg/DB01336.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00565.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01226.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOC1<br><img src="motifsvgg/motif_6928.svg" style="zoom:60%;" /></td>
    <td align="left">DB01421<br>DB00452</td>
    <td align="center"><img src="smilessvg/DB01421.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 41: #Drug1 may increase the hypotensive and central nervous system depressant (CNS depressant) activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00454</td>
    <td align="center"><img src="smilessvg/DB00454.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB00614<br>DB09245</td>
    <td align="center"><img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /></td>
    <td align="left">DB00805<br>DB01171</td>
    <td align="center"><img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 42: The risk or severity of hyperkalemia can be increased when #Drug1 is combined with #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /></td>
    <td align="left">DB00384<br>DB00594</td>
    <td align="center"><img src="smilessvg/DB00384.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00594.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /></td>
    <td align="left">DB00091</td>
    <td align="center"><img src="smilessvg/DB00091.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCOCCNCCC2CCC[C@@H](CCCC\C=C\C1)O2<br><img src="motifsvgg/motif_11033.svg" style="zoom:60%;" /></td>
    <td align="left">DB00864</td>
    <td align="center"><img src="smilessvg/DB00864.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 43: The protein binding of #Drug2 can be decreased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1=CNCSC1<br><img src="motifsvgg/motif_10059.svg" style="zoom:60%;" /></td>
    <td align="left">DB01327</td>
    <td align="center"><img src="smilessvg/DB01327.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1nncs1<br><img src="motifsvgg/motif_7505.svg" style="zoom:60%;" /></td>
    <td align="left">DB01327</td>
    <td align="center"><img src="smilessvg/DB01327.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCN1<br><img src="motifsvgg/motif_5848.svg" style="zoom:60%;" /></td>
    <td align="left">DB00252</td>
    <td align="center"><img src="smilessvg/DB00252.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 44: #Drug1 may increase the central neurotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB09245<br>DB00614</td>
    <td align="center"><img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /></td>
    <td align="left">DB00805<br>DB01171</td>
    <td align="center"><img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccnnc1<br><img src="motifsvgg/motif_5390.svg" style="zoom:60%;" /></td>
    <td align="left">DB00805</td>
    <td align="center"><img src="smilessvg/DB00805.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 45: #Drug1 may decrease effectiveness of #Drug2 as a diagnostic agent.</b></td>
  </tr>
  <tr>
    <td align="center">C1C[C@@H]2CCC[C@H]1N2<br><img src="motifsvgg/motif_4999.svg" style="zoom:60%;" /></td>
    <td align="left">DB08824<br>DB00907</td>
    <td align="center"><img src="smilessvg/DB08824.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00907.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01149<br>DB00490<br>DB00543</td>
    <td align="center"><img src="smilessvg/DB01149.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00490.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00543.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00813<br>DB00422<br>DB06701</td>
    <td align="center"><img src="smilessvg/DB00813.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06701.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 46: #Drug1 may increase the bronchoconstrictory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01136<br>DB08807</td>
    <td align="center"><img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08807.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB09204</td>
    <td align="center"><img src="smilessvg/DB09204.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCCC1<br><img src="motifsvgg/motif_6858.svg" style="zoom:60%;" /></td>
    <td align="left">DB04861</td>
    <td align="center"><img src="smilessvg/DB04861.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 48: #Drug1 may increase the myopathic rhabdomyolysis activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1c[n+]ccn1<br><img src="motifsvgg/motif_9727.svg" style="zoom:60%;" /></td>
    <td align="left">DB09055</td>
    <td align="center"><img src="smilessvg/DB09055.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /></td>
    <td align="left">DB00227</td>
    <td align="center"><img src="smilessvg/DB00227.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB08860</td>
    <td align="center"><img src="smilessvg/DB08860.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 49: The risk or severity of adverse effects can be increased when #Drug1 is combined with #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB09286<br>DB00454<br>DB01002</td>
    <td align="center"><img src="smilessvg/DB09286.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01002.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00363<br>DB01224<br>DB00298</td>
    <td align="center"><img src="smilessvg/DB00363.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01224.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00298.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00295<br>DB00497<br>DB00844</td>
    <td align="center"><img src="smilessvg/DB00295.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00497.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00844.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 50: The risk or severity of heart failure can be increased when #Drug2 is combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB06203<br>DB08882</td>
    <td align="center"><img src="smilessvg/DB06203.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08882.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCN1<br><img src="motifsvgg/motif_599.svg" style="zoom:60%;" /></td>
    <td align="left">DB00491<br>DB00419</td>
    <td align="center"><img src="smilessvg/DB00491.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00419.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB08907</td>
    <td align="center"><img src="smilessvg/DB08907.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 51: #Drug1 may increase the hypercalcemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /></td>
    <td align="left">DB00774</td>
    <td align="center"><img src="smilessvg/DB00774.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cnoc1<br><img src="motifsvgg/motif_3177.svg" style="zoom:60%;" /></td>
    <td align="left">DB01406</td>
    <td align="center"><img src="smilessvg/DB01406.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">S<br><img src="motifsvgg/motif_12299.svg" style="zoom:60%;" /></td>
    <td align="left">DB01021<br>DB00232<br>DB01324</td>
    <td align="center"><img src="smilessvg/DB01021.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00232.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01324.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 52: #Drug1 may decrease the analgesic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB11699<br>DB00757</td>
    <td align="center"><img src="smilessvg/DB11699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00757.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1ccCCN1<br><img src="motifsvgg/motif_3602.svg" style="zoom:60%;" /></td>
    <td align="left">DB00377</td>
    <td align="center"><img src="smilessvg/DB00377.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /></td>
    <td align="left">DB11699</td>
    <td align="center"><img src="smilessvg/DB11699.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 53: #Drug1 may increase the antiplatelet activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00235<br>DB05266</td>
    <td align="center"><img src="smilessvg/DB00235.svg" style="zoom:50%;" /><br><img src="smilessvg/DB05266.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /></td>
    <td align="left">DB01296</td>
    <td align="center"><img src="smilessvg/DB01296.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /></td>
    <td align="left">DB01240</td>
    <td align="center"><img src="smilessvg/DB01240.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 54: #Drug1 may increase the bradycardic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01136<br>DB06791<br>DB08877</td>
    <td align="center"><img src="smilessvg/DB01136.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06791.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08877.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /></td>
    <td align="left">DB00390<br>DB00511<br>DB01078</td>
    <td align="center"><img src="smilessvg/DB00390.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00511.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00390<br>DB01092<br>DB01078</td>
    <td align="center"><img src="smilessvg/DB00390.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01092.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 55: #Drug1 may increase the hyponatremic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cCOC1<br><img src="motifsvgg/motif_5314.svg" style="zoom:60%;" /></td>
    <td align="left">DB00215<br>DB01175</td>
    <td align="center"><img src="smilessvg/DB00215.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01175.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /></td>
    <td align="left">DB00774</td>
    <td align="center"><img src="smilessvg/DB00774.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCO1<br><img src="motifsvgg/motif_613.svg" style="zoom:60%;" /></td>
    <td align="left">DB00715</td>
    <td align="center"><img src="smilessvg/DB00715.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 56: The risk or severity of hypotension can be increased when #Drug1 is combined with #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cnccn1<br><img src="motifsvgg/motif_10902.svg" style="zoom:60%;" /></td>
    <td align="left">DB00594<br>DB00384</td>
    <td align="center"><img src="smilessvg/DB00594.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB11106<br>DB00384</td>
    <td align="center"><img src="smilessvg/DB11106.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /></td>
    <td align="left">DB00700<br>DB01395</td>
    <td align="center"><img src="smilessvg/DB00700.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01395.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 57: #Drug1 may increase the nephrotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /></td>
    <td align="left">DB00091</td>
    <td align="center"><img src="smilessvg/DB00091.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCOCCNCCC2CCC[C@@H](CCCC\C=C\C1)O2<br><img src="motifsvgg/motif_11033.svg" style="zoom:60%;" /></td>
    <td align="left">DB00864<br>DB00337</td>
    <td align="center"><img src="smilessvg/DB00864.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00337.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCOCC1<br><img src="motifsvgg/motif_8438.svg" style="zoom:60%;" /></td>
    <td align="left">DB01172<br>DB00452<br>DB00684</td>
    <td align="center"><img src="smilessvg/DB01172.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00452.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00684.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 58: #Drug1 may decrease the cardiotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /></td>
    <td align="left">DB01092<br>DB01078<br>DB01396</td>
    <td align="center"><img src="smilessvg/DB01092.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /></td>
    <td align="left">DB00511<br>DB01078<br>DB01396</td>
    <td align="center"><img src="smilessvg/DB00511.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB06771<br>DB08911<br>DB00537</td>
    <td align="center"><img src="smilessvg/DB06771.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08911.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00537.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 59: #Drug1 may increase the ulcerogenic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /></td>
    <td align="left">DB00572<br>DB00245<br>DB00424</td>
    <td align="center"><img src="smilessvg/DB00572.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00424.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOccC1<br><img src="motifsvgg/motif_908.svg" style="zoom:60%;" /></td>
    <td align="left">DB00940<br>DB00782</td>
    <td align="center"><img src="smilessvg/DB00940.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00782.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00340<br>DB00967</td>
    <td align="center"><img src="smilessvg/DB00340.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 60: #Drug1 may increase the hypotensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1=CNC=CC1<br><img src="motifsvgg/motif_7812.svg" style="zoom:60%;" /></td>
    <td align="left">DB09236<br>DB00528<br>DB06712</td>
    <td align="center"><img src="smilessvg/DB09236.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00528.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06712.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCN1<br><img src="motifsvgg/motif_9508.svg" style="zoom:60%;" /></td>
    <td align="left">DB00599<br>DB00474<br>DB01351</td>
    <td align="center"><img src="smilessvg/DB00599.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00474.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01351.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00734<br>DB00346<br>DB08932</td>
    <td align="center"><img src="smilessvg/DB00734.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00346.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08932.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 61: #Drug1 may decrease the stimulatory activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB09128<br>DB00831<br>DB01624</td>
    <td align="center"><img src="smilessvg/DB09128.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00831.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01624.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB01100<br>DB06144<br>DB04842</td>
    <td align="center"><img src="smilessvg/DB01100.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06144.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04842.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /></td>
    <td align="left">DB01614<br>DB00433<br>DB01608</td>
    <td align="center"><img src="smilessvg/DB01614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00433.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 62: The bioavailability of #Drug2 can be increased when combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /></td>
    <td align="left">DB01200<br>DB01253<br>DB00353</td>
    <td align="center"><img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00353.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00950<br>DB00699<br>DB00248</td>
    <td align="center"><img src="smilessvg/DB00950.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00248.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB01200<br>DB01186</td>
    <td align="center"><img src="smilessvg/DB01200.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01186.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 63: #Drug1 may increase the myelosuppressive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /></td>
    <td align="left">DB01348<br>DB00492<br>DB00584</td>
    <td align="center"><img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00492.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1c[nH]cn1<br><img src="motifsvgg/motif_11664.svg" style="zoom:60%;" /></td>
    <td align="left">DB01033</td>
    <td align="center"><img src="smilessvg/DB01033.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00440</td>
    <td align="center"><img src="smilessvg/DB00440.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 64: #Drug1 may increase the serotonergic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cc[nH]c1<br><img src="motifsvgg/motif_6846.svg" style="zoom:60%;" /></td>
    <td align="left">DB00150<br>DB11699<br>DB00757</td>
    <td align="center"><img src="smilessvg/DB00150.svg" style="zoom:50%;" /><br><img src="smilessvg/DB11699.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00757.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB09042<br>DB00601<br>DB09245</td>
    <td align="center"><img src="smilessvg/DB09042.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00601.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB09209<br>DB00854<br>DB00652</td>
    <td align="center"><img src="smilessvg/DB09209.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00854.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00652.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 66: The risk or severity of bleeding can be increased when #Drug1 is combined with #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB12364<br>DB09075<br>DB00608</td>
    <td align="center"><img src="smilessvg/DB12364.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09075.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccsc1<br><img src="motifsvgg/motif_4969.svg" style="zoom:60%;" /></td>
    <td align="left">DB06228<br>DB00744<br>DB01600</td>
    <td align="center"><img src="smilessvg/DB06228.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00744.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01600.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1=CNScc1<br><img src="motifsvgg/motif_2638.svg" style="zoom:60%;" /></td>
    <td align="left">DB00814<br>DB00554<br>DB06725</td>
    <td align="center"><img src="smilessvg/DB00814.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00554.svg" style="zoom:50%;" /><br><img src="smilessvg/DB06725.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 68: #Drug1 may increase the hyperkalemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /></td>
    <td align="left">DB01395<br>DB00700</td>
    <td align="center"><img src="smilessvg/DB01395.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00700.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00440<br>DB01349<br>DB00384</td>
    <td align="center"><img src="smilessvg/DB00440.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01349.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00384.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /></td>
    <td align="left">DB01348<br>DB09477<br>DB00584</td>
    <td align="center"><img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09477.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 69: #Drug1 may increase the analgesic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cC2CCN[C@H](C1)C2<br><img src="motifsvgg/motif_1215.svg" style="zoom:60%;" /></td>
    <td align="left">DB00611<br>DB01466<br>DB00652</td>
    <td align="center"><img src="smilessvg/DB00611.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00652.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cOCC1<br><img src="motifsvgg/motif_8789.svg" style="zoom:60%;" /></td>
    <td align="left">DB01466<br>DB01551<br>DB00318</td>
    <td align="center"><img src="smilessvg/DB01466.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01551.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00318.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB06738<br>DB01081<br>DB00454</td>
    <td align="center"><img src="smilessvg/DB06738.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01081.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00454.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 70: The therapeutic efficacy of #Drug2 can be decreased when used in combination with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB06698<br>DB01132<br>DB00608</td>
    <td align="center"><img src="smilessvg/DB06698.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01132.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00608.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB06148<br>DB00334<br>DB08815</td>
    <td align="center"><img src="smilessvg/DB06148.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00334.svg" style="zoom:50%;" /><br><img src="smilessvg/DB08815.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00358<br>DB09097<br>DB00843</td>
    <td align="center"><img src="smilessvg/DB00358.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09097.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00843.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 71: #Drug1 may increase the hypertensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1=CCNCC1<br><img src="motifsvgg/motif_11627.svg" style="zoom:60%;" /></td>
    <td align="left">DB00696<br>DB01253<br>DB00353</td>
    <td align="center"><img src="smilessvg/DB00696.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01253.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00353.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB09245<br>DB00614<br>DB00601</td>
    <td align="center"><img src="smilessvg/DB09245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00601.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00248<br>DB00320<br>DB00422</td>
    <td align="center"><img src="smilessvg/DB00248.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00320.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00422.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 74: #Drug1 may increase the fluid retaining activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCCCO1<br><img src="motifsvgg/motif_6908.svg" style="zoom:60%;" /></td>
    <td align="left">DB00621</td>
    <td align="center"><img src="smilessvg/DB00621.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cn[nH]c1<br><img src="motifsvgg/motif_980.svg" style="zoom:60%;" /></td>
    <td align="left">DB06718</td>
    <td align="center"><img src="smilessvg/DB06718.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cnoc1<br><img src="motifsvgg/motif_3177.svg" style="zoom:60%;" /></td>
    <td align="left">DB01406</td>
    <td align="center"><img src="smilessvg/DB01406.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 76: #Drug1 may decrease the sedative activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB01176<br>DB04841<br>DB00354</td>
    <td align="center"><img src="smilessvg/DB01176.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04841.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00354.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00719<br>DB00455<br>DB00920</td>
    <td align="center"><img src="smilessvg/DB00719.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00455.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00920.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00792<br>DB00719<br>DB00967</td>
    <td align="center"><img src="smilessvg/DB00792.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00719.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00967.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 78: #Drug1 may increase the hyperglycemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /></td>
    <td align="left">DB00288</td>
    <td align="center"><img src="smilessvg/DB00288.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">S<br><img src="motifsvgg/motif_12299.svg" style="zoom:60%;" /></td>
    <td align="left">DB09063</td>
    <td align="center"><img src="smilessvg/DB09063.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">O=S<br><img src="motifsvgg/motif_5204.svg" style="zoom:60%;" /></td>
    <td align="left">DB09063</td>
    <td align="center"><img src="smilessvg/DB09063.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 79: #Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1NCCO1<br><img src="motifsvgg/motif_9993.svg" style="zoom:60%;" /></td>
    <td align="left">DB00614<br>DB09245</td>
    <td align="center"><img src="smilessvg/DB00614.svg" style="zoom:50%;" /><br><img src="smilessvg/DB09245.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1COCCN1<br><img src="motifsvgg/motif_5448.svg" style="zoom:60%;" /></td>
    <td align="left">DB00805<br>DB01171</td>
    <td align="center"><img src="smilessvg/DB00805.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01171.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNCC1<br><img src="motifsvgg/motif_2890.svg" style="zoom:60%;" /></td>
    <td align="left">DB00813</td>
    <td align="center"><img src="smilessvg/DB00813.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 80: #Drug1 may increase the hepatotoxic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00951<br>DB00908<br>DB00613</td>
    <td align="center"><img src="smilessvg/DB00951.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00908.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00613.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCNCCNCCNCCNCCNCCNCCNCCNCCNCCN1<br><img src="motifsvgg/motif_950.svg" style="zoom:60%;" /></td>
    <td align="left">DB00091</td>
    <td align="center"><img src="smilessvg/DB00091.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00619<br>DB01254<br>DB00943</td>
    <td align="center"><img src="smilessvg/DB00619.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01254.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00943.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 81: #Drug1 may increase the thrombogenic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1ccCN1<br><img src="motifsvgg/motif_1707.svg" style="zoom:60%;" /></td>
    <td align="left">DB01041<br>DB00480</td>
    <td align="center"><img src="smilessvg/DB01041.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00480.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCCN1<br><img src="motifsvgg/motif_599.svg" style="zoom:60%;" /></td>
    <td align="left">DB01041<br>DB00480</td>
    <td align="center"><img src="smilessvg/DB01041.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00480.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCCO1<br><img src="motifsvgg/motif_7036.svg" style="zoom:60%;" /></td>
    <td align="left">DB01395</td>
    <td align="center"><img src="smilessvg/DB01395.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 82: #Drug1 may increase the arrhythmogenic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">C1CCCOC1<br><img src="motifsvgg/motif_10853.svg" style="zoom:60%;" /></td>
    <td align="left">DB01078<br>DB01396<br>DB00511</td>
    <td align="center"><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00511.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1C=CCO1<br><img src="motifsvgg/motif_11215.svg" style="zoom:60%;" /></td>
    <td align="left">DB01078<br>DB01396<br>DB01092</td>
    <td align="center"><img src="smilessvg/DB01078.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01396.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01092.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cSccN1<br><img src="motifsvgg/motif_5078.svg" style="zoom:60%;" /></td>
    <td align="left">DB01071<br>DB00679<br>DB01608</td>
    <td align="center"><img src="smilessvg/DB01071.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00679.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01608.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 83: #Drug1 may increase the hypokalemic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cSNCN1<br><img src="motifsvgg/motif_2693.svg" style="zoom:60%;" /></td>
    <td align="left">DB00774</td>
    <td align="center"><img src="smilessvg/DB00774.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00214</td>
    <td align="center"><img src="smilessvg/DB00214.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1OCCO1<br><img src="motifsvgg/motif_1065.svg" style="zoom:60%;" /></td>
    <td align="left">DB00288<br>DB00273</td>
    <td align="center"><img src="smilessvg/DB00288.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00273.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 84: #Drug1 may increase the vasopressor activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cNccCC1<br><img src="motifsvgg/motif_7515.svg" style="zoom:60%;" /></td>
    <td align="left">DB01242<br>DB00726<br>DB00458</td>
    <td align="center"><img src="smilessvg/DB01242.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00726.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00458.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CNCCN1<br><img src="motifsvgg/motif_3795.svg" style="zoom:60%;" /></td>
    <td align="left">DB00370<br>DB00543</td>
    <td align="center"><img src="smilessvg/DB00370.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00543.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1ccncc1<br><img src="motifsvgg/motif_110.svg" style="zoom:60%;" /></td>
    <td align="left">DB00370</td>
    <td align="center"><img src="smilessvg/DB00370.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 85: #Drug1 may increase the tachycardic activities of #Drug2.</b></td>
  </tr>
  <tr>
    <td align="center">c1cOCCC1<br><img src="motifsvgg/motif_6858.svg" style="zoom:60%;" /></td>
    <td align="left">DB00486<br>DB00470<br>DB04861</td>
    <td align="center"><img src="smilessvg/DB00486.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00470.svg" style="zoom:50%;" /><br><img src="smilessvg/DB04861.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1=NCCN1<br><img src="motifsvgg/motif_10330.svg" style="zoom:60%;" /></td>
    <td align="left">DB06711<br>DB00751<br>DB00964</td>
    <td align="center"><img src="smilessvg/DB06711.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00751.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00964.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1C[C@H]2CCC[C@@H]1N2<br><img src="motifsvgg/motif_7409.svg" style="zoom:60%;" /></td>
    <td align="left">DB00245<br>DB00424<br>DB00572</td>
    <td align="center"><img src="smilessvg/DB00245.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00424.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00572.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td colspan="3" align="left"><b>DDI Type 86: The risk of a hypersensitivity reaction to #Drug2 is increased when it is combined with #Drug1.</b></td>
  </tr>
  <tr>
    <td align="center">c1cncnc1<br><img src="motifsvgg/motif_3655.svg" style="zoom:60%;" /></td>
    <td align="left">DB00437</td>
    <td align="center"><img src="smilessvg/DB00437.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">c1cn[nH]c1<br><img src="motifsvgg/motif_980.svg" style="zoom:60%;" /></td>
    <td align="left">DB00437</td>
    <td align="center"><img src="smilessvg/DB00437.svg" style="zoom:50%;" /></td>
  </tr>
  <tr>
    <td align="center">C1CCNC1<br><img src="motifsvgg/motif_2958.svg" style="zoom:60%;" /></td>
    <td align="left">DB00722<br>DB01348<br>DB00584</td>
    <td align="center"><img src="smilessvg/DB00722.svg" style="zoom:50%;" /><br><img src="smilessvg/DB01348.svg" style="zoom:50%;" /><br><img src="smilessvg/DB00584.svg" style="zoom:50%;" /></td>
  </tr>
</table>



```Note: A motif is considered present in a drug if it appears among the top-ranked substructures identified by TRACE for that molecule.```