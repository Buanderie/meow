--[[ 
Copyright (c) 
]]--

require('torch')
require('nn')
require('optim')

function table.length(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

local clock = os.clock
function sleep(n)  -- seconds
   local t0 = clock()
   while clock() - t0 <= n do
   end
end

local dqa = torch.class('DeepQAgent')

function dqa:initNeuralNet()
	self.net = nn.Sequential()
	--self.net:add(nn.TemporalConvolution(
	self.net:add(nn.TemporalConvolution(3,16,5,1))
	self.net:add(nn.ReLU())
	self.net:add(nn.TemporalMaxPooling(2))
	self.net:add(nn.ReLU())
	self.net:add(nn.TemporalConvolution(16,32,5,1))
	self.net:add(nn.ReLU())
	self.net:add(nn.TemporalMaxPooling(2))
	
	local testInput = torch.randn( self.stock_input_len, 3 );
	local testOutput = self.net:forward( testInput )
	local viewSize = testOutput:size()[1] * testOutput:size()[2]
	
	self.net:add(nn.View(viewSize))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(viewSize, viewSize/2))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(viewSize/2, 128))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(128, self.number_of_actions))	

	self.criterion = nn.MSECriterion()
	
	self.parameters, self.gradParameters = self.net:getParameters()
end

function dqa:saveNetwork()
	torch.save('net.bin', self.net)
end

function dqa:loadNetwork()
	self.net = torch.load( 'net.bin' )
	self.criterion = nn.MSECriterion()
	self.parameters, self.gradParameters = self.net:getParameters()
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
	self.stock_input_len = 16

	--- current number of iteration	
	self.iter = 0

	--- epsilon annealing
	self.ep_start = 	0.95
	self.ep	=			self.ep_start
	self.ep_end =		0.000001
	self.ep_end_t =		1000000

	--- replay memory
	--- max size of replay memory
	self.replay_memory_max_size = 100000
	--- actual replay memory
	self.replay_memory = {}
	
	--- Training
	--- Training batch size
	self.training_batch_size = 200
   	self.learning_rate = 0.1
   	self.learning_rate_decay = 5e-7
   	self.momentum = 0.9
   	self.coefL1 = 0.001
   	self.coefL2 = 0.001
   	
	--- Gamma 
	-- gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
   	-- Determines the amount of weight placed on the utility of the state resulting from an action.
   	self.gamma = 0.9
   	
	--- neural net
	--- initialized to random weights if no params
	if args.agent_net ~= nil then
		--- load agent neural net
		self:loadNetwork()
	else
		self:initNeuralNet()	
	end
end

function dqa:actRandom( input )
	t = torch.Tensor(1)
	t:random(1,3)
	return t
end

function dqa:forward( input )
	local ret = self.net:forward( input )
	return ret
end

function dqa:policy( input )
	local action_values = self:forward( input )
	local maxval = action_values[1]
  	local max_index = 1
 
 	-- find maximum output and note its index and value
  	for i = 2, self.number_of_actions do
  		if action_values[i] > maxval then
  			maxval = action_values[i]
  			max_index = i
  		end
  	end
  
  	return {action = max_index, value = maxval};
end

function dqa:actFromNet( input )
	print("############## ACT FROM BRAIN ################")
	local ret = self:policy( input )	
	--print( ret )
	print( "############# END ###########################")
	return ret.action
end

function dqa:insertToMemory( tuple )	
	--- if we've reached the maximum number of experiences in memory, remove one at random place (like amnesia)
	if table.length( self.replay_memory ) > self.replay_memory_max_size - 1 then
		local sampleIdx = math.random( 1, table.length(self.replay_memory))

		table.remove( self.replay_memory, sampleIdx )
	end 
	
	table.insert( self.replay_memory, tuple )
end

function dqa:trainFromMemory()

	inputs = torch.Tensor(self.training_batch_size, self.stock_input_len, 3 )
	targets = torch.Tensor(self.training_batch_size, self.number_of_actions, 1 )
		
	print( "Training with " .. tostring( self.training_batch_size ) .. " samples" )
	
	for k = 1, self.training_batch_size do
		--- Choose tuple randomly from replay memory
		local sampleIdx = math.random( 1, table.length(self.replay_memory))
		local sample = self.replay_memory[ sampleIdx ];
		-- print( sample )
		
		-- copy state from experience S0
        local x = torch.Tensor(sample[1]);
   
   		-- compute best action for the new state S1
        local best_action = self:policy(sample[4]);
        
        --[[ get current action output values
   				we want to make the target outputs the same as the actual outputs
   				expect for the action that was chose - we want to replace this with
	   			the reward that was obtained + the utility of the resulting state
   			--]]
   			
   		
   		local all_outputs = self.net:forward(x);
		inputs[k] = x:clone();      	
		targets[k] = all_outputs:clone();
		targets[k][2] = sample[3] + self.gamma * best_action.value; 
	end
	
		--- EVAL CLOSURE STARTS
		-- create training function to give to optim.sgd
		local feval = function(x)
	     collectgarbage()

	     -- get new network parameters
	     if x ~= self.parameters then
	        self.parameters:copy(x)
	     end

	     -- reset gradients
	     self.gradParameters:zero()

	     -- evaluate function for complete mini batch
	     local outputs = self.net:forward(inputs)
	     local f = self.criterion:forward(outputs, targets)

	     -- estimate df/dW
	     local df_do = self.criterion:backward(outputs, targets)
	     self.net:backward(inputs, df_do)

	     -- penalties (L1 and L2):
	     if self.coefL1 ~= 0 or Brain.coefL2 ~= 0 then
	        -- locals:
	       local norm,sign = torch.norm,torch.sign

	        -- Loss:
	        f = f + self.coefL1 * norm(self.parameters,1)
	        f = f + self.coefL2 * norm(self.parameters,2)^2/2

	        -- Gradients:
	        self.gradParameters:add( sign(self.parameters):mul(self.coefL1) + self.parameters:clone():mul(self.coefL2) )
	     end

	     -- return f and df/dX
	     return f, self.gradParameters
	  	end
	  	--- EVAL CLOSURE ENDS
	  	
	  	-- fire up optim.sgd
		sgdState = {
            learningRate = self.learning_rate,
            momentum = self.momentum,
            learningRateDecay = self.learning_rate_decay
         }
         
        _,fs = optim.sgd(feval, self.parameters, sgdState)
        print( "Current loss: " .. tostring(fs[1] ) )
        -- sleep(1)
        
end

function dqa:train( stepTuple )
	
	--- Insert tuple to memory
	self:insertToMemory( stepTuple )
	
	--- Train
	if table.length( self.replay_memory ) > self.training_batch_size then
		self:trainFromMemory()
	end
	
end

function dqa:actOnInput( input )

	print( self.ep )

	--- epsilon greedy
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
