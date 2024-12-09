import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.utils import shuffle
import string
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel, AlbertTokenizer, TFAlbertModel
import tensorflow as tf
import time
from datetime import datetime
from Data.data_process import extract_variable_from_log
from graph import graph_builder as gb
from graph_emb.models import SDNE,DeepWalk,node2vec
from graph_emb import utils
from tqdm import tqdm
import mmh3
# Pre-trained BERT model
bert_tokenizer = BertTokenizer.from_pretrained('log/BERT')
bert_model = TFBertModel.from_pretrained('log/BERT')

roberta_tokenizer = RobertaTokenizer.from_pretrained('log/RoBERTa')
roberta_model = TFRobertaModel.from_pretrained('log/RoBERTa')

albert_tokenizer = AlbertTokenizer.from_pretrained('log/ALBERT')
albert_model = TFAlbertModel.from_pretrained("log/ALBERT")

def bert_encoder(s, no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, padding=True, truncation=True, max_length=512, return_tensors="tf")
    outputs = bert_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]

def robert_encoder(s,no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = roberta_tokenizer(s, padding=True, truncation=True, max_length=512, return_tensors="tf")
    outputs = roberta_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]

def albert_encoder(s, no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in albert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = albert_tokenizer(s, padding=True, truncation=True, max_length=512, return_tensors="tf")
    outputs = albert_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

def _split_data(x_data, y_data=None, train_ratio=0.8, split_type='uniform'):
    """ Split train/test data
    Parameters
    ----------
    x_data: list, set of log sequences (in the type of semantic vectors)
    y_data: list, labels for each log sequence
    train_ratio: float, training ratio (e.g., 0.8)
    split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
    Returns
    -------

    """
    (x_data, y_data) = shuffle(x_data, y_data)
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = train_pos
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)


def graph_embed(G):
    line = {}
    embed_dict = {}
    for node_id in tqdm(G.nodes()):
        line[node_id] = G.nodes[node_id]['line']
    # model = SDNE(G)
    # model = DeepWalk(G, walk_length=5, num_walks=10, workers=3)
    model = node2vec.Node2Vec(G, walk_length = 5, num_walks = 5)
    model.train()
    embeddings = model.get_embeddings()
    tensor_dict = {key: (tf.convert_to_tensor(value),line[key])for key, value in embeddings.items()}
    for k,v in tensor_dict.items():
        tensor = v[0]
        data = v[1] 
        for i in data:
            if i not in embed_dict.keys():
                embed_dict[i] = [tensor]
            else:
                embed_dict[i].append(tensor)
    return  embed_dict

def hash_func(attribute, dim=768):
    hash_value = mmh3.hash(attribute)% (2**32 - 1)
    # hash_value = hash(attribute)% (2**32 - 1)
    np.random.seed(hash_value)
    return np.random.rand(dim)

def attr_embed(G):
    attribute = ['size','packetresponse','srcPort','dstPort']
    attr_dict = {}
    embeddings = {}
    for node in G.nodes():
        node_attrs = G.nodes[node]
        node_embeddings = {}
        for attr_key, attr_value in reversed(node_attrs.items()):
            if attr_key in attribute:
                last_line_idx = len(node_attrs['line']) - 1
                last_line_value = node_attrs['line'][last_line_idx]
                
                for value in reversed(attr_value):
                    attr_embedding = hash_func(value)
                    node_embeddings[attr_key] = (tf.convert_to_tensor(attr_embedding, dtype=tf.float32), last_line_value)
                    if last_line_idx > 0:
                        last_line_idx -= 1

        embeddings[node] = node_embeddings
    for k,v in embeddings.items():
        for sub_k,sub_v in v.items():
            i = sub_v[-1]
            if i not in attr_dict.keys():
                attr_dict[i] = [sub_v[0]]
            else:
                attr_dict[i].append(sub_v[0])
    return attr_dict

def ablation(data):
    embeddings = {}
    repeat = {}
    for item in data:
        if(item[0] not in repeat.keys()):
            hash_value = tf.convert_to_tensor(hash_func(item[0]), dtype=tf.float32)
            repeat[item[0]] = hash_value
        else:
            hash_value = repeat[item[0]]
        if item[2] not in embeddings.keys():
            embeddings[item[2]] = [hash_value]
        else:
            embeddings[item[2]].append(hash_value)
    return embeddings

def load_data(log_file, label_file = None,train_ratio=0.5,split_type='uniform',no_wordpiece = 0):
    E = {}
    blk_dict = {}
    blk_tensor = {}
    t0 = time.time()
    assert log_file.endswith('.log'), "Missing .log file"
    # elif log_file.endswith('.log'):
    print("Loading", log_file)
    with open(log_file,'r') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    data_dict = OrderedDict()
    n_logs = len(logs)
    print(n_logs)
    print("Loaded", n_logs, "lines!")

    data = extract_variable_from_log(log_file)
    print("variables extraction done!")
    G = gb.create_attribute_graph(data)
    print("graph construction done!")
    gembeddings = graph_embed(G)
    aembeddings = attr_embed(G)
    # print(gembeddings)
    # print(aembeddings)
    for k,v in aembeddings.items():
        gembeddings[k] = gembeddings[k] + v
    # print(gembeddings)
    for i, line in enumerate(logs):
        timestamp = " ".join(line.split()[:2])
        timestamp = datetime.strptime(timestamp, '%y%m%d %H%M%S').timestamp()
        blkId_list = re.findall(r'(blk_-?\d+)', line)
        blkId_list = list(set(blkId_list))
        if len(blkId_list) >= 2:
            continue
        blkId_set = set(blkId_list)
        content = clean(line).lower()
        if content not in E.keys():
            # E[content] = bert_encoder(content, no_wordpiece)
            # E[content] = robert_encoder(content,no_wordpiece)
            E[content] = albert_encoder(content,no_wordpiece)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append((E[content], timestamp))
            if not blk_Id in blk_dict:
                blk_dict[blk_Id] = [i]
            else:
                blk_dict[blk_Id].append(i)
        # print(data_dict)
        i += 1
        if i % 1000 == 0 or i == n_logs:
            print("\rLoading {0:.2f}% - number of unique message: {1}".format(i / n_logs * 100, len(E.keys())), end="")
    for k,v in blk_dict.items():
        for i in v:
            if k not in blk_tensor.keys():
                blk_tensor[k] = gembeddings[i]
            else:
                blk_tensor[k] = blk_tensor[k] + gembeddings[i]
    for k, v in data_dict.items():
        seq = [x[0] for x in v]
        rt = [x[1] for x in v]
        rt = [rt[i] - rt[i - 1] for i in range(len(rt))]
        rt = [0] + rt
        data_dict[k] = (seq, rt)
    data_df = [(k, v[0], v[1]) for k, v in data_dict.items()]
    for i,item in enumerate(data_df):
        k = item[0]
        v = item[1:]
        data_df[i] = (k,v[0] + blk_tensor[k],v[1])
    data_df = pd.DataFrame(data_df, columns=['BlockId', 'EventSequence', "TimeSequence"])
    
    if label_file:
        # Split training and validation set in a class-uniform way
        label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        print("\n")
        print("Saving data...")
        # data_df.to_csv("output.csv", index=True)
        np.savez_compressed("data-{0}.npz".format('bert'), data_x=data_df['EventSequence'].values,
                            data_y=data_df['Label'].values)
        # Split train and test data
        (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type)

        print(y_train.sum(), y_test.sum())
    else:
        raise NotImplementedError("Missing label file for the HDFS dataset!")
    
    if label_file is None:
        if split_type == 'uniform':
            split_type = 'sequential'
            print('Warning: Only split_type=sequential is supported \
            if label_file=None.'.format(split_type))
        # Split training and validation set sequentially
        x_data = data_df['EventSequence'].values
        (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
        print('Total: {} instances, train: {} instances, test: {} instances'.format(
            x_data.shape[0], x_train.shape[0], x_test.shape[0]))
        return (x_train, None), (x_test, None), data_df
    # else:
    #     raise NotImplementedError('load_HDFS() only support csv and npz files!')
    print("\nLoaded all HDFS dataset in: ", time.time() - t0)

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)

def ab_load_data(log_file, label_file = None,train_ratio=0.5,split_type='uniform',no_wordpiece = 0):
    E = {}
    blk_dict = {}
    blk_tensor = {}
    t0 = time.time()
    assert log_file.endswith('.log'), "Missing .log file"
    # elif log_file.endswith('.log'):
    print("Loading", log_file)
    with open(log_file,'r') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    data_dict = OrderedDict()
    n_logs = len(logs)
    print(n_logs)
    print("Loaded", n_logs, "lines!")

    data = extract_variable_from_log(log_file)
    print(len(data))
    print("variables extraction done!")
    abla_embeddings = ablation(data)
    print(len(abla_embeddings))
    # print(gembeddings)
    # print(aembeddings)
    # print(gembeddings)
    for i, line in enumerate(logs):
        timestamp = " ".join(line.split()[:2])
        timestamp = datetime.strptime(timestamp, '%y%m%d %H%M%S').timestamp()
        blkId_list = re.findall(r'(blk_-?\d+)', line)
        blkId_list = list(set(blkId_list))
        if len(blkId_list) >= 2:
            continue
        blkId_set = set(blkId_list)
        content = clean(line).lower()
        if content not in E.keys():
            # E[content] = bert_encoder(content, no_wordpiece)
            # E[content] = robert_encoder(content,no_wordpiece)
            E[content] = albert_encoder(content,no_wordpiece)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append((E[content], timestamp))
            if not blk_Id in blk_dict:
                blk_dict[blk_Id] = [i]
            else:
                blk_dict[blk_Id].append(i)
        # print(data_dict)
        i += 1
        if i % 1000 == 0 or i == n_logs:
            print("\rLoading {0:.2f}% - number of unique message: {1}".format(i / n_logs * 100, len(E.keys())), end="")
    for k,v in blk_dict.items():
        for i in v:
            if k not in blk_tensor.keys():
                blk_tensor[k] = abla_embeddings[i]
            else:
                blk_tensor[k] = blk_tensor[k] + abla_embeddings[i]
    for k, v in data_dict.items():
        seq = [x[0] for x in v]
        rt = [x[1] for x in v]
        rt = [rt[i] - rt[i - 1] for i in range(len(rt))]
        rt = [0] + rt
        data_dict[k] = (seq, rt)
    data_df = [(k, v[0], v[1]) for k, v in data_dict.items()]
    for i,item in enumerate(data_df):
        k = item[0]
        v = item[1:]
        data_df[i] = (k,v[0] + blk_tensor[k],v[1])
    data_df = pd.DataFrame(data_df, columns=['BlockId', 'EventSequence', "TimeSequence"])
    
    if label_file:
        # Split training and validation set in a class-uniform way
        label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        print("\n")
        print("Saving data...")
        # data_df.to_csv("output.csv", index=True)
        np.savez_compressed("data-{0}.npz".format('bert'), data_x=data_df['EventSequence'].values,
                            data_y=data_df['Label'].values)
        # Split train and test data
        (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type)

        print(y_train.sum(), y_test.sum())
    else:
        raise NotImplementedError("Missing label file for the HDFS dataset!")
    
    if label_file is None:
        if split_type == 'uniform':
            split_type = 'sequential'
            print('Warning: Only split_type=sequential is supported \
            if label_file=None.'.format(split_type))
        # Split training and validation set sequentially
        x_data = data_df['EventSequence'].values
        (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
        print('Total: {} instances, train: {} instances, test: {} instances'.format(
            x_data.shape[0], x_train.shape[0], x_test.shape[0]))
        return (x_train, None), (x_test, None), data_df
    # else:
    #     raise NotImplementedError('load_HDFS() only support csv and npz files!')
    print("\nLoaded all HDFS dataset in: ", time.time() - t0)

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_tr, y_tr), (x_te, y_te) = load_data(
        "./Data/test.log", "./Data/anomaly_label.csv", train_ratio=0.8, split_type='sequential')
    # print(len(x_tr),len(y_tr))