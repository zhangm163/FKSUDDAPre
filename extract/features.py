from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
import timeit
from joblib import Parallel, delayed


class DfVec(object):
    def __init__(self, vec):
        self.vec = vec
        if type(self.vec) != np.ndarray:
            raise TypeError('numpy.ndarray expected, got %s' % type(self.vec))

    def __str__(self):
        return "%s dimensional vector" % str(self.vec.shape)

    __repr__ = __str__

    def __len__(self):
        return len(self.vec)

    _repr_html_ = __str__


class MolSentence:

    def __init__(self, sentence):
        self.sentence = sentence
        if type(self.sentence[0]) != str:
            raise TypeError('List with strings expected')

    def __len__(self):
        return len(self.sentence)

    def __str__(self):
        return 'MolSentence with %i words' % len(self.sentence)

    __repr__ = __str__

    def contains(self, word):
        if word in self.sentence:
            return True
        else:
            return False

    __contains__ = contains

    def __iter__(self):
        for x in self.sentence:
            yield x

    _repr_html_ = __str__



def mol2sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}
    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element
    identifier_sentences = []

    for r in radii:
        identifiers = []
        for atom in dict_atoms:
            identifiers.append(dict_atoms[atom][r])
        identifier_sentences.append(list(map(str, [x for x in identifiers if x])))
    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(identifier_sentences), list(alternating_sentence)


def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element
    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def _parallel_job(mol, r):
    """Helper function for joblib jobs
    """
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, r)
        return " ".join(sentence)


def _read_smi(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield Chem.MolFromSmiles(line.split('\t')[0])


def generate_corpus(in_file, out_file, r, sentence_type='alt', n_jobs=1):

    # File type detection
    in_split = in_file.split('.')
    if in_split[-1].lower() not in ['sdf', 'smi', 'ism', 'gz']:
        raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    gzipped = False
    if in_split[-1].lower() == 'gz':
        gzipped = True
        if in_split[-2].lower() not in ['sdf', 'smi', 'ism']:
            raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')

    file_handles = []

    # write only files which contain corpus
    if (sentence_type == 'individual') or (sentence_type == 'all'):

        f1 = open(out_file+'_r0.corpus', "w")
        f2 = open(out_file+'_r1.corpus', "w")
        file_handles.append(f1)
        file_handles.append(f2)

    if (sentence_type == 'alt') or (sentence_type == 'all'):
        f3 = open(out_file, "w")
        file_handles.append(f3)

    if gzipped:
        import gzip
        if in_split[-2].lower() == 'sdf':
            mols_file = gzip.open(in_file, mode='r')
            suppl = Chem.ForwardSDMolSupplier(mols_file)
        else:
            mols_file = gzip.open(in_file, mode='rt')
            suppl = _read_smi(mols_file)
    else:
        if in_split[-1].lower() == 'sdf':
            suppl = Chem.ForwardSDMolSupplier(in_file)
        else:
            mols_file = open(in_file, mode='rt')
            suppl = _read_smi(mols_file)

    if sentence_type == 'alt':  # This can run parallelized
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(mol, r) for mol in suppl)
        for i, line in enumerate(result):
            f3.write(str(line) + '\n')
        print('% molecules successfully processed.')

    else:
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles)
                identifier_sentences, alternating_sentence = mol2sentence(mol, r)

                identifier_sentence_r0 = " ".join(identifier_sentences[0])
                identifier_sentence_r1 = " ".join(identifier_sentences[1])
                alternating_sentence_r0r1 = " ".join(alternating_sentence)

                if len(smiles) != 0:
                    if (sentence_type == 'individual') or (sentence_type == 'all'):
                        f1.write(str(identifier_sentence_r0)+'\n')
                        f2.write(str(identifier_sentence_r1)+'\n')

                    if (sentence_type == 'alt') or (sentence_type == 'all'):
                        f3.write(str(alternating_sentence_r0r1)+'\n')

    for fh in file_handles:
        fh.close()


def _read_corpus(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield line.split()


def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK'):
    f = open(corpus)
    unique = {}
    for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
        for identifier in x:
            if identifier not in unique:
                unique[identifier] = 1
            else:
                unique[identifier] += 1
    n_lines = i + 1
    least_common = set([x for x in unique if unique[x] <= threshold])
    f.close()

    f = open(corpus)
    fw = open(out_corpus, mode='w')
    for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
        intersection = set(line) & least_common
        if len(intersection) > 0:
            new_line = []
            for item in line:
                if item in least_common:
                    new_line.append(uncommon)
                else:
                    new_line.append(item)
            fw.write(" ".join(new_line) + '\n')
        else:
            fw.write(" ".join(line) + '\n')
    f.close()
    fw.close()


def train_word2vec_model(infile_name, outfile_name=None, vector_size=100, window=10, min_count=3, n_jobs=1,
                         method='skip-gram', **kwargs):
    if method.lower() == 'skip-gram':
        sg = 1
    elif method.lower() == 'cbow':
        sg = 0
    else:
        raise ValueError('skip-gram or cbow are only valid options')

    start = timeit.default_timer()
    corpus = word2vec.LineSentence(infile_name)
    model = word2vec.Word2Vec(corpus, size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg,
                              **kwargs)
    if outfile_name:
        model.save(outfile_name)

    stop = timeit.default_timer()
    print('Runtime: ', round((stop - start)/60, 2), ' minutes')
    return model


def remove_salts_solvents(smiles, hac=3):
    save = []
    for el in smiles.split("."):
        mol = Chem.MolFromSmiles(str(el))
        if mol.GetNumHeavyAtoms() <= hac:
            save.append(mol)

    return ".".join([Chem.MolToSmiles(x) for x in save])


def sentences2vec(sentences, model, unseen=None):
    keys = set(model.wv.key_to_index.keys())
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


def featurize(in_file, out_file, model_path, r, uncommon=None):
    # Load the model
    word2vec_model = word2vec.Word2Vec.load(model_path)
    if uncommon:
        try:
            word2vec_model.wv[uncommon]
        except KeyError:
            raise KeyError('Selected word for uncommon: %s not in vocabulary' % uncommon)

    # File type detection
    in_split = in_file.split('.')
    f_type = in_split[-1].lower()
    if f_type not in ['sdf', 'smi', 'ism', 'gz']:
        raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    if f_type == 'gz':
        if in_split[-2].lower() not in ['sdf', 'smi', 'ism']:
            raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
        else:
            f_type = in_split[-2].lower()

    print('Loading molecules.')
    if f_type == 'sdf':
        df = PandasTools.LoadSDF(in_file)
        print("Keeping only molecules that can be processed by RDKit.")
        df = df[df['ROMol'].notnull()]
        df['Smiles'] = df['ROMol'].map(Chem.MolToSmiles)
    else:
        df = pd.read_csv(in_file, delimiter='\t', usecols=[0, 1], names=['Smiles', 'ID'])
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Smiles')
        print("Keeping only molecules that can be processed by RDKit.")
        df = df[df['ROMol'].notnull()]
        df['Smiles'] = df['ROMol'].map(Chem.MolToSmiles)  

    print('Featurizing molecules.')
    df['mol2vec_6000_down_sampling-sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], r)), axis=1)
    vectors = sentences2vec(df['mol2vec_6000_down_sampling-sentence'], word2vec_model, unseen=uncommon)
    df_vec = pd.DataFrame(vectors, columns=['mol2vec_5000_down_sampling-%03i' % x for x in range(vectors.shape[1])])
    df_vec.index = df.index
    df = df.join(df_vec)

    df.drop(['ROMol', 'mol2vec_6000_down_sampling-sentence'], axis=1).to_csv(out_file)
