--[[ 
Copyright (c) 3015 Monadic Labs
]]--

require('torch')

local dqa = torch.class('DeepQAgent')

function dqa:__init(args)
	
	--- data input
	self.stock_input_len = 128
	
	--- epsilon annealing
	self.ep_start = 	1
	self.ep	=		self.ep_start
	self.ep_end =		0.000001
	self.ep_end_t =		1000000

	--- replay memory
	--- max size of replay memory
	self.replay_memory_max_size = 1000

	
end
