require('torch')
require('nn')

module = nn.JoinTable(2, 2)

x = torch.randn(3, 1)
y = torch.randn(3, 1)

mx = torch.randn(2, 3, 1)
my = torch.randn(2, 3, 1)

print(module:forward{x, y})
print(module:forward{mx, my})

--[[
input = torch.rand(16, 3, 1) -- simulating 3 channels
n_feature_maps = 10

print(input:size())
print(input)

local parconv = nn.Parallel(2,1)
	for i = 1,3 do -- I need to generate 3 sequential structures
    	local model = nn.Sequential()
    	model:add(nn.TemporalConvolution(1,16,5))
    	model:add(nn.ReLU())
    	model:add(nn.TemporalMaxPooling(2))
    	model:add(nn.TemporalConvolution(16,16,5))
    	model:add(nn.ReLU())
    	model:add(nn.TemporalMaxPooling(2))
    	model:add(nn.ReLU())
    	parconv:add(model) -- add each to the main model
	end
	
output = parconv:forward(input)
print( output )
--------

root = nn.Sequential()
conv_part = nn.Identity()
id_part = nn.Identity()
mlp = nn.ParallelTable()
mlp:add(conv_part)
mlp:add(id_part)
root:add(mlp)
root:add(nn.JoinTable(1))

x = torch.randn(10)
y = torch.rand(5)
caca = {{x, y}, {x, y}}
--popo = root:forward( {{x, y}, {x, y}} )
print( caca )
--print( popo )
]]--

x1 = torch.rand(16, 1) -- or x1 = torch.rand(10) for individual inputs
--x1 = torch.rand(16, 10, 1) -- or x1 = torch.rand(10) for individual inputs
x2 = torch.rand(3, 2) -- or x2 = torch.rand(10) for individual inputs
--x2 = torch.rand(16, 2) -- or x2 = torch.rand(10) for individual inputs

model1 = nn.Sequential()
model1:add(nn.TemporalConvolution(1,4,5,1))
--model1:add(nn.Reshape(32))
-- model1:add(nn.ReLU())
-- model1:add(nn.TemporalMaxPooling(2))
-- model1:add(nn.ReLU())
-- m = nn.View(-1):setNumInputDims(2)
-- model1:add(m)

-- model1:add(nn.View(28))

--model1:add(nn.TemporalConvolution(16,32,5,1))
--model1:add(nn.ReLU())
--model1:add(nn.TemporalMaxPooling(2))

model2 = nn.Sequential():add(nn.Linear(2, 1)):add(nn.Tanh())
model3 = nn.Linear(29, 4)
global = nn.Sequential():add(nn.ParallelTable():add(model1):add(model2)):add(nn.JoinTable(1, 1)):add(model3)

--pred = global:forward({x1, x2})
print( "pred1" )
pred1 = model1:forward(x1)
print( pred1 )
print( "pred2" )
pred2 = model2:forward(x2)

print(pred1)
print(pred2)

pred = global:forward( {x1, x2} )
print(pred)

--pred = model2:forward(x2)
--print(pred)
--pred = global:forward({x1, x2})
-- print(pred)
