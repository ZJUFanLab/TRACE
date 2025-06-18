import numpy as np
import torch
import dgl
import pandas as pd
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import Draw

from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import pickle

eletype_list = [i for i in range(118)]

hrc2emb = {}
for eletype in eletype_list:
    hrc_emb = np.random.rand(14)
    hrc2emb[eletype] = hrc_emb
def hrc_features(ele):
    fhrc = hrc2emb[ele]
    return fhrc.tolist()


rel2emb = pickle.load(open('data/elementkg/rel2emb.pkl','rb'))
def relation_features(e1,e2):
    frel = rel2emb[(e1,e2)]
    return frel.tolist()

# 加载 ele2emb.pkl 文件
ele2emb_file_path = 'data/elementkg/ele2emb.pkl'  
with open(ele2emb_file_path, 'rb') as file:
    ele2emb = pickle.load(file)

# 加载 fg2emb.pkl 文件
fg2emb_file_path = 'data/elementkg/fg2emb.pkl'  
with open(fg2emb_file_path, 'rb') as file:
    fg2emb = pickle.load(file)

# 加载官能团信息
with open('data/elementkg/funcgroup.txt', 'r') as file:
    funcgroups = file.read().strip().split('\n')
    smart = [Chem.MolFromSmarts(line.split()[1]) for line in funcgroups]
    name = [line.split()[0] for line in funcgroups]
    smart2name = dict(zip(smart, name))

def match_fg(mol):
    fg_emb = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            fg_emb.append(fg2emb[smart2name[sm]].tolist())
    if len(fg_emb) > 13:
        fg_emb = fg_emb[:13]
    else:
        fg_emb.extend([[0] * 133] * (13 - len(fg_emb)))
    return fg_emb


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def get_atom_features(atom, molecule):
    possible_atom = [
        'C', 'O', 'N', 'Ti', 'S', 'Cl', 'Br', 'I', 'K', 'Bi', 'Al', 'Mg', 
        'Ca', 'Si', 'Na', 'P', 'F', 'Pt', 'B', 'Li', 'Gd', 'As', 'Tc', 'Fe', 
        'Ga', 'H', 'Ag', 'Ra', 'La', 'Sr', 'Zn', 'Cu', 'Se', 'Cr', 'Au', 
        'Hg', 'Co', 'Sb','Mn','Lu'
    ]
    symbol = atom.GetSymbol()
    if symbol not in possible_atom:
        print("Encountered unknown atom type:", symbol)
        symbol = 'DU'  # 'DU' 用作未知原子类型的占位符
    atom_features = one_of_k_encoding_unk(symbol, possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), 
                                           [Chem.rdchem.HybridizationType.SP, 
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3, 
                                            Chem.rdchem.HybridizationType.SP3D])
    atomic_num = atom.GetAtomicNum()
    if atomic_num in ele2emb:
        atom_embedding = ele2emb[atomic_num]
    else:
        atom_embedding = [0] * len(next(iter(ele2emb.values())))

    # 将嵌入向量添加到原子特征中
    atom_features.extend(atom_embedding)
    return np.array(atom_features)


def bond_features(bond):
    # 初始化特征列表
    bond_feats = []

    # 添加键类型特征
    bond_type = bond.GetBondType()
    bond_feats.extend([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC
    ])

    # 添加其他特征：是否共轭、是否在环中
    bond_feats.append(bond.GetIsConjugated())
    bond_feats.append(bond.IsInRing())

    return np.array(bond_feats)


def get_bond_features(molecule, i, j, rel2emb, hrc2emb):
    bond = molecule.GetBondBetweenAtoms(i, j)
    if bond is None:
        return None
    # 使用标准键特征
    f_bond = bond_features(bond)
    
    return f_bond


def get_bond_features_self_loop():
    bond_features_self_loop = np.zeros(6, dtype=np.float32)
    
    return bond_features_self_loop


def smiles_to_dgl(smiles_str):
    molecule = Chem.MolFromSmiles(smiles_str)

    if molecule is None:
        return None
    
    G = dgl.graph(([], []))
    G.add_nodes(molecule.GetNumAtoms())
    
    node_features = []
    edge_features = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i, molecule) 
        node_features.append(atom_i_features)
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i,j) 
                bond_features_ij = get_bond_features(molecule, i, j, rel2emb, hrc2emb)
                edge_features.append(bond_features_ij)
    G.ndata['atom'] = torch.FloatTensor(np.array(node_features))
    if G.number_of_edges() == 0:
        G = dgl.add_self_loop(G)
        for i in range(G.num_nodes()):
            bond_features_self_loop = get_bond_features_self_loop()
            edge_features.append(bond_features_self_loop)
    else:
        G.edata['bond'] = torch.FloatTensor(np.array(edge_features))
    if G.number_of_edges() > 0:
        edge_feats = np.array(edge_features)
        if edge_feats.ndim == 1 and edge_feats.shape[0] == 6:
            G.edata['bond'] = torch.FloatTensor(edge_feats).view(1, -1)
        elif edge_feats.ndim == 2 and edge_feats.shape[1] == 6:
            G.edata['bond'] = torch.FloatTensor(edge_feats)
        else:
            print("Problematic SMILES:", smiles_str)
            print("Edge features shape:", edge_feats.shape)
    else:
        print("No edges in molecule, SMILES:", smiles_str)

    G = dgl.add_self_loop(G)
    return G


def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    # print(len(smiles))
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            mol = Chem.MolToSmiles(mol)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
            print(len(graphs))
        except:
            success.append(False)
    # print(len(graphs))

    return graphs, success
if __name__ == '__main__':
    g = smiles_to_dgl("NC(=O)C1=CC=CC=C1O")
    print(g)
    if g is not None:
        print("Generated Graph:", g)
        # 打印每个节点的特征
        for node_id in range(g.number_of_nodes()):
            node_feat = g.ndata['atom'][node_id]
            print(f"Node {node_id} features:", node_feat)
        # 打印每个边的特征
        for src, dst in zip(*g.edges()):
            edge_feat = g.edata['bond'][g.edge_ids(src, dst)]
            print(f"Edge from {src.item()} to {dst.item()} features:", edge_feat)
    else:
        print("Failed to generate a graph from the provided SMILES string.")
    atom_embedding_len = len(next(iter(ele2emb.values())))
    print(f"原子嵌入向量的长度是: {atom_embedding_len}")

