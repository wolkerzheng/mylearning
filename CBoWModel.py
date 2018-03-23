#encoding=utf8

import torch 
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self,vocab_size,context,embedding_dim):
        super(CBOW,self).__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.linear = nn.Linear(embedding_dim,vocab_size)
        pass

    def forward(self, inputs):
        embds = self.embedding(inputs)
        sumEm = torch.sum(embds,dim=0).view(1,-1)
        out = F.log_softmax(self.linear(sumEm),dim=1)
        return out

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


make_context_vector(data[0][0], word_to_ix)  # example

model = CBOW(vocab_size,CONTEXT_SIZE,embedding_dim=30)
loss_function = nn.NLLLoss()
losses = []
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(30):
    total_loss = torch.Tensor([0])
    for context, target  in data:

        context_vec = make_context_vector(context,word_to_ix)
        label_vec = autograd.Variable(torch.LongTensor(word_to_ix[target]))

        model.zero_grad()
        log_probs = model(context_vec)

        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        loss.backward()
        optimizer.step()


        total_loss += loss.data
    losses.append(total_loss)
print(losses)