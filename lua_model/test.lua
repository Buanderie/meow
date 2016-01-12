require('torch')
require('nn')

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
