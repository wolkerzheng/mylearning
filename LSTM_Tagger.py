#encoding=utf8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)
"""

{'Everybody': 5, 'ate': 2, 'apple': 4, 'that': 7, 'read': 6, 'dog': 1, 'book': 8, 'the': 3, 'The': 0}
{'DET': 0, 'NN': 1, 'V': 2}
('before:', Variable containing:
-0.9717 -1.1357 -1.2027
-0.9890 -1.1333 -1.1839
-0.8769 -1.1833 -1.2814
-0.8363 -1.0987 -1.4551
-0.8968 -1.2236 -1.2108
[torch.FloatTensor of size 5x3]
)
('after', Variable containing:
-0.1691 -2.2880 -2.9169
-5.7522 -0.0093 -5.1071
-3.7977 -3.4605 -0.0553
-0.0511 -3.5692 -3.8330
-4.0823 -0.0237 -5.0265
[torch.FloatTensor of size 5x3]
)


"""
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
EPOCHS = 300

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
	]
# print(training_data.shape)
def load_word_to_idx(data):
	word_to_idx = {}
	tag_to_idx = {}
	for words,tags in training_data:
		for word in words:
			if word not in word_to_idx:
				word_to_idx[word] = len(word_to_idx)
		for t in tags:
			if t not in tag_to_idx:
				tag_to_idx[t] = len(tag_to_idx)


	return word_to_idx,tag_to_idx


def prepare_sequence(seq,to_ix):
	idxs = [to_ix[w] for w in seq]
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)


class LSTMTagger(nn.Module):
	"""docstring for LSTMTagger"""
	def __init__(self, embedding_dim,hidden_dim,vocab_size,tagset_size):
		super(LSTMTagger, self).__init__()
		# self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		
		self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
		#输入是词向量的维数，输出是隐藏状态
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		# 隐藏层：将lstm的输出作为输入，然后输出标签
		self.hidden2tag = nn.Linear(hidden_dim,tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		#the axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
			autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

	def forward(self,sentence):
		embed = self.word_embeddings(sentence)
		lstm_out,self.hidden = self.lstm(embed.view
			(len(sentence),1,-1),self.hidden)
		tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
		tag_scores = F.log_softmax(tag_space,dim=1)
		return tag_scores


word_to_idx,tag_to_idx =  load_word_to_idx(training_data)
print(word_to_idx)
print(tag_to_idx)


model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_idx),len(tag_to_idx))

loss_function = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(),lr=0.1)

#学习模型前
inputs = prepare_sequence(training_data[0][0],word_to_idx)
tag_scores = model(inputs)
print('before:',tag_scores)

for epoch in range(EPOCHS):

	for sentence,tags in training_data:
		#pytorch会累加梯度，所以在每个例子前需要更新置为0
		model.zero_grad()
		#同样的，需要清除隐藏层状态
		model.hidden = model.init_hidden()

		#数据处理
		sentence_in = prepare_sequence(sentence,word_to_idx)
		targets = prepare_sequence(tags,tag_to_idx)

		#运行forward
		tag_scores = model(sentence_in)

		#计算损失梯度，并且更新参数
		loss = loss_function(tag_scores,targets)

		loss.backward()
		optimizer.step()
#学习模型后
inputs = prepare_sequence(training_data[0][0], word_to_idx)
tag_scores = model(inputs)
print('after',tag_scores)
