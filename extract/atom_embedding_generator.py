import os
import numpy as np
import torch
from smiles import construct_input_from_smiles
from my_nn import EarlyStopping, set_random_seed, BERT_atom_embedding_generator

set_random_seed()

def bert_atom_embedding(smiles, pretrain_model='pretrain_k_bert_epoch_7.pth'):

    args = {
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'metric_name': 'roc_auc',
        'batch_size': 128,
        'num_epochs': 200,
        'd_model': 768,
        'n_layers': 6,
        'vocab_size': 47,
        'maxlen': 201,
        'd_k': 64,
        'd_v': 64,
        'd_ff': 768 * 4,
        'n_heads': 12,
        'global_labels_dim': 1,
        'atom_labels_dim': 15,
        'lr': 3e-5,
        'pretrain_layer': 6,
        'mode': 'higher',
        'task_name': 'HIA',
        'patience': 20,
        'times': 10,
        'pretrain_model': pretrain_model,
    }


    token_idx,atom_mask_list = construct_input_from_smiles(smiles)
    print(f"token_idx: {token_idx}")
    print(f"atom_mask_list: {atom_mask_list}")

    model = BERT_atom_embedding_generator(
        d_model=args['d_model'],
        n_layers=args['n_layers'],
        vocab_size=args['vocab_size'],
        maxlen=args['maxlen'],
        d_k=args['d_k'],
        d_v=args['d_v'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        global_label_dim=args['global_labels_dim'],
        atom_label_dim=args['atom_labels_dim'],
        use_atom=False
    )
    stopper = EarlyStopping(
        pretrained_model=args['pretrain_model'],
        pretrain_layer=args['pretrain_layer'],
        mode=args['mode']
    )
    model.to(args['device'])


    try:
        stopper.load_pretrained_model(model)
        print("模型加载成功")
    except Exception as e:
        print(f"加载预训练模型时出错: {e}")
        raise e


    token_idx_tensor = torch.tensor([token_idx]).long().to(args['device'])
    atom_mask_np = np.array(atom_mask_list)
    atom_mask_index = np.where(atom_mask_np == 1)
    print(f"atom_mask_index: {atom_mask_index}")
    print(f"atom_mask_index size: {atom_mask_index[0].size}")
    h_global, h_atom = model(token_idx_tensor, atom_mask_index)
    h_global = h_global.cpu().squeeze().detach().numpy()
    h_atom = h_atom.cpu().squeeze().detach().numpy()
    print(f"h_global shape: {h_global.shape}")
    print(f"h_atom shape: {h_atom.shape}")
    print(f"device: {args['device']}")
    print(f"token_idx_tensor: {token_idx_tensor}")
    print(f"atom_mask: {atom_mask_list}")

    return h_global, h_atom
