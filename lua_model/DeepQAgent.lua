--[[ 
Copyright (c) 
]]--

require('torch')
require('nn')

local dqa = torch.class('DeepQAgent')

function dqa:initNeuralNet()
	self.net = nn.Sequential()
	self.net:add(nn.TemporalConvolution(1,16,5,1))
	self.net:add(nn.TemporalMaxPooling(2))
	self.net:add(nn.TemporalConvolution(16,32,5,1))
	self.net:add(nn.TemporalMaxPooling(2))
	self.net:add(nn.View(1952))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(1952, 1952/2))
	self.net:add(nn.Tanh())
	self.net:add(nn.Linear(1952/2, 128))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(128, self.number_of_actions))	
	print(self.net)
end

function dqa:chooseAction(state)
	ret = self.net:forward(state)
	return ret
end

function dqa:__init(args)
	
	--- agent model
	--- 3 actions : buy one, sell one, none
	self.number_of_actions = 3
	
	--- data input
	self.stock_input_len = 256

	--- current number of iteration	
	self.iter = 0

	--- epsilon annealing
	self.ep_start = 	1
	self.ep	=		self.ep_start
	self.ep_end =		0.000001
	self.ep_end_t =		1000000

	--- replay memory
	--- max size of replay memory
	self.replay_memory_max_size = 1000

	--- neural net
	--- initialized to random weights if no params
	if args.agent_net ~= nil then
		--- load agent neural net
	else
		self:initNeuralNet()	
	end
end

function dqa:actRandom( input )
	t = torch.Tensor(1)
	t:random(1,3)
	return t
end

function dqa:actFromNet( input )
	ret = self.net:forward( input )
	y,i=torch.max(ret,1)
	return i
end

function dqa:actOnInput( input )

	print( self.ep )

	--- epsilon annealing
	rr = torch.uniform()
	local ret = nil
	if rr < self.ep then
		ret = self:actRandom(input)
	else
		ret = self:actFromNet(input)
	end

	-- anneal the epsilon a little
	self.ep = self.ep - 0.001

	-- return choosen action
	return ret
end
