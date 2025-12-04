
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens
def construct_input_from_smiles(smiles, max_len=200):
    try:
        # built a pretrain data from smiles
        atom_list = []
        atom_token_list = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',
                           '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']
        all_token_list = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']

        word2idx = {}
        for i, w in enumerate(all_token_list):
            word2idx[w] = i
        token_list = smi_tokenizer(smiles)
        if len(token_list) > max_len:
            token_list = token_list[:max_len]
        tokens = ['[GLO]'] + token_list
        padding_list = ['[PAD]' for x in range(max_len - len(token_list))]
        tokens += padding_list
        atom_mask_list = []
        atom_labels_list = []
        index = 0
        tokens_idx = []
        for i, token in enumerate(tokens):
            if token in atom_token_list:
                atom_mask_list.append(1)
                index = index + 1
                tokens_idx.append(word2idx[token])
            else:
                if token in all_token_list:
                    tokens_idx.append(word2idx[token])
                    atom_mask_list.append(0)
                elif '[' in list(token):
                    tokens[i] = '[other_atom]'
                    atom_mask_list.append(1)
                    index = index + 1
                    tokens_idx.append(word2idx['[other_atom]'])
                else:
                    tokens[i] = '[other_token]'
                    tokens_idx.append(word2idx['[other_token]'])
                    atom_mask_list.append(0)


        tokens_idx = [word2idx[x] for x in tokens]
        if len(tokens_idx) == max_len + 1:
            return tokens_idx, atom_mask_list
        else:
            return 0, 0
    except:
        return 0, 0


def extract_middle_smiles(smiles, length=200):
    if len(smiles) > length:
        start_idx = (len(smiles) - length) // 2
        return smiles[start_idx:start_idx + length]
    return smiles

def check(smiles):
    ls = smi_tokenizer(smiles)
    if(len(ls) > 200):
        return False
    else:
        return True
    

def cut_smiles(smiles):
    if(check(smiles) == False):
        for i in range(200, 0, -1):
            new_smiles = extract_middle_smiles(smiles)
            if(check(new_smiles) == True):
                return new_smiles
        return ""
    return smiles

def bert_atom_embedding(smiles):
    token_idx, atom_mask_list = construct_input_from_smiles(smiles)
    token_idx = torch.tensor([token_idx]).long().to(args['device'])
    atom_mask = atom_mask_list
    atom_mask_np = np.array(atom_mask)
    atom_mask_index = np.where(atom_mask_np == 1)
    h_global, h_atom = drug_model(token_idx, atom_mask_index)
    h_global = h_global.cpu().squeeze().detach().numpy()
    h_atom = h_atom.cpu().squeeze().detach().numpy()
    return h_global, h_atom

def process_smiles(smiles):
    smiles = cut_smiles(smiles)
    return bert_atom_embedding(smiles)