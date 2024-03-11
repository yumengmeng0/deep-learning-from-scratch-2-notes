import sys

sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen()
model.load_params('../ch06/BetterRnnlm.pkl')

# 设定start字符和skip字符
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 文本生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)

model.reset_state()

start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-'*50)
print(txt)

'''
！！！生成文本中含有$
--------------------------------------------------
the meaning of life is that it is n't taking as an investment operation said barry gray director of the state of economic affairs.
 according to to investors events in the latest fiscal year the markets are too small to attract.
 most people are still predicting the or the incomplete accounts of the securities industry rather than edge.
 but at last week 's earthquake oct..
 equaling c$ the london bank petroleum iron projects and the rest of the century mentality the pent
'''