import torch
x = torch.empty(5,3)
print(x)

x = torch.zeros(5,3,dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5,3,dtype=torch.float)
print(x)

x = torch.randn_like(x)
print(x)

print(x.size())
print(x.shape)














