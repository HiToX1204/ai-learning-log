import torch
import torch.nn as nn
x=torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y=torch.tensor([[0.], [1.], [1.], [0.]])
model=nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
)
loss_fn=nn.MSELoss()
loss_fn(model(x),y)
opt=torch.optim.SGD(model.parameters(),lr=0.1)
for i in range(5000):
    opt.zero_grad()
    pred=model(x)
    loss=loss_fn(pred,y)
    loss.backward()
    opt.step()
print("loss=",round(loss.item(),6))
print("pred=",[round(v,3) for v in pred.flatten().tolist()])
print("y    =",y.flatten().tolist())
