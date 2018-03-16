#encoding=utf8
import torch 
from torch.autograd import Variable
import torch.nn as nn 
class TwoLayerNet(nn.Module):

	def __init__(self,D_in,H,D_out):

		super(TwoLayerNet,self).__init__()
		self.linear1 = nn.Linear(D_in,H)
		self.linear2 = nn.Linear(H,D_out)

	def forward(self,input):
		"""
		input data is Variable
		return data :Variable

		"""
		h_relu = self.linear1(input).clamp(min=0)	
		y_pred = self.linear2(h_relu)
		return y_pred


batch_size,D_in,H,D_out = 64,1000,100,10
NUM_RANGE = 500

x = Variable(torch.randn(batch_size,D_in))
y = Variable(torch.randn(batch_size,D_out),requires_grad=False)


model = TwoLayerNet(D_in,batch_size,D_out)

cirterion = nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)

for t in range(NUM_RANGE):

	y_pred = model(x)

	loss = cirterion(y_pred,y)

	print(t,loss.data[0])
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	