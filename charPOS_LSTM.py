#encoding=utf8

import torch
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
torch.manual_seed(1)
"""
Augmenting the LSTM part-of-speech tagger with character-level features
"""
WORD_EMBEDDING = 20
CHAR_EMBEDDING = 10
MAX_WORD_LEN = 10
HIDDEN_DIM = 6
CHAR_HIDDEN_DIM = 5
training_data = [
	("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]


# char_to_ix = {}
# tag_to_ix = {}

# char_to_ix = {c:i for c,i in enumerate('a'+26)}

def load_word_to_idx(data):
	word_to_idx = {}
	tag_to_idx = {}
	char_to_ix = {}
	char_to_ix[' '] = len(char_to_ix)
	for words,tags in training_data:
		for word in words:
			if word not in word_to_idx:
				word_to_idx[word] = len(word_to_idx)
			for ch in word:
				if ch not in char_to_ix:
					char_to_ix[ch] = len(char_to_ix)
		for t in tags:
			if t not in tag_to_idx:
				tag_to_idx[t] = len(tag_to_idx)


	return word_to_idx,tag_to_idx,char_to_ix

def prepare_sequence(seq,to_ix):

	# res = 
	# for w in seq:
	idxs = [ to_ix[w] for w in seq]
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)

class LSTMTagger(nn.Module):

	def __init__(self,char_size,vocab_size,target_size,char_embedding_dim,
			word_embedding_dim,hidden_dim,out_char_dim):
		super(LSTMTagger,self).__init__()
		
		self.hidden_dim = hidden_dim
		self.out_char_dim = out_char_dim

		self.charembedding = nn.Embedding(char_size,char_embedding_dim)

		self.wordembedding = nn.Embedding(vocab_size,word_embedding_dim)

		self.lstm1 = nn.LSTM(char_embedding_dim,out_char_dim)

		self.lstm2 = nn.LSTM(out_char_dim+word_embedding_dim,hidden_dim)

		self.hidden2tag  = nn.Linear(hidden_dim,target_size)

		self.hidden = self.init_hidden()

	def init_hidden(self):

		return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
			autograd.Variable(torch.zeros(1,1,self.hidden_dim))) 
	def inti_hidden_char(self,words_num):

		return (autograd.Variable(torch.zeros(1,words_num,self.out_char_dim)),
			autograd.Variable(torch.zeros(1,words_num,self.out_char_dim)))
	def forward(self,sentence,char_seq_list,max_word=10):

		sentence_size = sentence.size()[0]  
		word_embed = self.wordembedding(sentence)
		char_embed = self.charembedding(char_seq_list)
		
		# 
		# char_lstm_out,
		try:
			char_embed = char_embed.view(len(sentence),max_word,-1).permute(1,0,2)
		except:
			print('char_embed.size:',char_embed.size())

		self.hidden_char = self.inti_hidden_char(sentence_size)
		char_lst,self.inti_hidden_char = self.lstm1(char_embed,self.inti_hidden_char)
		char_embeds = char_lst[-1,:,:].view(sentence_size,-1)  

		embed = torch.cat((word_embed,char_embeds),dim=1)
		self.hidden = self.init_hidden()
		lstm_out, self.hidden = self.lstm2(embed.view(sentence_size,1,-1), self.hidden) 
		tag_space = self.hidden2tag(lstm_out.view(sentence_size,-1))  
		tag_scores = F.log_softmax(tag_space)  
		return tag_scores  

class CharLSTM(nn.Module):
	def __init__(self,char_size,char_embedding_dim,hidden_dim):
		super(CharLSTM,self).__init__()
		self.embedding = nn.Embedding(char_size,char_embedding_dim)
		self.lstm = nn.LSTM(char_embedding_dim,hidden_dim)

		self.hidden = hidden_dim
	def init_hidden():
		return (autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
			autograd.Variable(torch.zeros(1,1,self.hidden_dim)))
		pass
	def forward(self,input_words):
		embds = self.embedding(input_words)

		out = self.lstm(embds.view(1,len(input_words),-1),self.hidden_dim)
		return out

word_to_idx,tag_to_idx,char_to_ix = load_word_to_idx(training_data)
# def train():
model = LSTMTagger(len(char_to_ix),len(word_to_idx), len(tag_to_idx), 
	CHAR_EMBEDDING,WORD_EMBEDDING,  HIDDEN_DIM,CHAR_HIDDEN_DIM)  
loss_function = nn.NLLLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.1) 


for epoch in range(30):

	for sentence,tags in training_data:

		model.zero_grad()
		model.hidden = model.init_hidden()  
		sentence_in = prepare_sequence(sentence,word_to_idx)
		targets = prepare_sequence(tags,tag_to_idx)
		# char_in = torch.zeros()
		sent_chars = []
		for word in sentence:
			w = ' '*(MAX_WORD_LEN-len(word))
			sent_chars.extend(list(word + w) if len(word)<MAX_WORD_LEN else list(word[:MAX_WORD_LEN]))  
		char_in = prepare_sequence(word,char_to_ix)

		tag_pred = model(sentence_in,char_in,MAX_WORD_LEN)
		loss = loss_function(tag_pred,targets)
		loss.backward()
		optimizer.step()

