import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
import os
import pickle
import pandas as pd


flatten = lambda l: [item for sublist in l for item in sublist]

USE_CUDA = torch.cuda.is_available()
torch.cuda.set_device(0)

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))


label_path = '../../../media/bach3/seungheon/MSD_allMusicTags/ground_truth_assignments/AMG_Multilabel_tagsets'
label_name = ['msd_amglabels_all.h5','msd_amglabels_genres.h5','msd_amglabels_moods.h5','msd_amglabels_styles.h5','msd_amglabels_themes.h5']
types ='all'
save_path = '../dataset'
type_dict = {
        'all' : 0,
        'genres' : 1,
        'moods' : 2,
        'styles' : 3,
        'themes' : 4
        }

df = pd.read_hdf(os.path.join(label_path, label_name[type_dict[types]]))
shuffle_corpus = []
for i in range(len(df)):
    temp = []
    for j in df.iloc[i]:
        if type(j) is not float:
            temp.extend(j)
            random.shuffle(temp)
    shuffle_corpus.append(temp)


corpus = [[word.lower() for word in sent] for sent in shuffle_corpus][0:25000]
print(len(corpus), corpus[0])


word_count = Counter(flatten(corpus))
MIN_COUNT = 3
exclude = []
for w, c in word_count.items():
    if c < MIN_COUNT:
        exclude.append(w)

vocab = list(set(flatten(corpus)) - set(exclude))

word2index = {}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
        
index2word = {v:k for k, v in word2index.items()}

WINDOW_SIZE = 5
windows =  flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

train_data = []

for window in windows:
    for i in range(WINDOW_SIZE * 2 + 1):
        if window[i] in exclude or window[WINDOW_SIZE] in exclude: 
            continue # min_count
        if i == WINDOW_SIZE or window[i] == '<DUMMY>': 
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))

X_p = []
y_p = []

for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    y_p.append(prepare_word(tr[1], word2index).view(1, -1))
    
train_data = list(zip(X_p, y_p))

print("lenth of train_data",len(train_data))

Z = 0.001
word_count = Counter(flatten(corpus))
num_total_words = sum([c for w, c in word_count.items() if w not in exclude])

unigram_table = []

for vo in vocab:
    unigram_table.extend([vo] * int(((word_count[vo]/num_total_words)**0.75)/Z))

# Negative Sampling

def negative_sampling(targets, unigram_table, k):
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
        while len(nsample) < k: # num of sampling
            neg = random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))
    
    return torch.cat(neg_samples)


class SkipgramNegSampling(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim) # out embedding
        self.logsigmoid = nn.LogSigmoid()
                
        initrange = (2.0 / (vocab_size + projection_dim))**0.5 # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        
        neg_embeds = -self.embedding_u(negative_words) # B x K x D
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
        
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        
        return -torch.mean(loss)
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds


EMBEDDING_SIZE = 300
BATCH_SIZE = 128
EPOCH = 50
NEG = 10 # Num of Negative Sampling

losses = []
model = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
    print("Cuda using")
else:
    print("Cuda not using")
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCH):
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        
        inputs, targets = zip(*batch)
        
        inputs = torch.cat(inputs) # B x 1
        targets = torch.cat(targets) # B x 1
        negs = negative_sampling(targets, unigram_table, NEG)
        model.zero_grad()

        loss = model(inputs, targets, negs)
        
        loss.backward()
        optimizer.step()
    
        losses.append(loss.data.tolist())
    if epoch % 10 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
        losses = []


def word_similarity(target, vocab):
    if USE_CUDA:
        target_V = model.prediction(prepare_word(target, word2index))
    else:
        target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target: 
            continue
        
        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

def save_vector(vocab):
    vectors = {}
    for i in range(len(vocab)):
        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
            token = list(vocab)[i]
            vectors[token] = vector
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
            token = list(vocab)[i]
            vectors[token] = vector

    with open('../dataset/SkipGram_Vec.pkl', 'wb') as f:
        pickle.dump(vectors, f)

    print("Finish Pickling")  

test = random.choice(list(vocab))
print('bittersweet', word_similarity('bittersweet', vocab))
print('breakup', word_similarity('breakup', vocab))

print('rainy day', word_similarity('rainy day', vocab))
print('stay in bed', word_similarity('stay in bed', vocab))
save_vector(vocab)