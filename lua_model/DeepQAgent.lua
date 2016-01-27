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

	-- self.net = nn.Sequential()
	
	--[[
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
	
	self.net:add( parconv )
	
	local testInput = torch.randn( self.stock_input_len, 3, 1 );
	--print(testInput)
	local testOutput = self.net:forward( testInput )
	local viewSize = testOutput:size()[1] * testOutput:size()[2]
	
	self.net:add( nn.Reshape( viewSize ) )
	
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(viewSize, viewSize/2))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(viewSize/2, 128))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(128, self.number_of_actions))
	]]--
	
	--self.net:add(nn.TemporalConvolution(
	--[[
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
	self.net:add(nn.ReLU())
	]]--
	
	model1 = nn.Sequential()
	
	model1:add( nn.TemporalConvolution(1,32,5,1) )
	model1:add( nn.Tanh() )
	model1:add( nn.TemporalMaxPooling(2) )
	model1:add( nn.TemporalConvolution(32,16,5,1) )
	model1:add( nn.Tanh() )
	model1:add( nn.TemporalMaxPooling(2) )
	
	-- model1:add( nn.Identity() )
	local m = nn.View(-1):setNumInputDims(2)
    model1:add(m)
        
	model2 = nn.Sequential():add( nn.Identity() )
	
	local size1 = model1:forward( torch.rand( self.stock_input_len, 1 ) ):size()
	size1 = size1[#size1]
	-- print( "size1:" )
	-- print( size1 )
	local size2 = model2:forward( torch.rand( 2 ) ):size()[1]
	local inSize = size1 + size2
	-- print( inSize )
	
	model3 = nn.Sequential()
	model3:add( nn.Linear( inSize, inSize * 2 ) )
	model3:add( nn.Tanh() )
	model3:add( nn.Linear( inSize * 2, inSize ) )
	model3:add( nn.Tanh() )
	model3:add( nn.Linear( inSize, self.number_of_actions ) )
	model3:add( nn.Tanh() )
	
	self.net = nn.Sequential():add(nn.ParallelTable():add(model1):add(model2)):add(nn.JoinTable(1, 1)):add(model3)
	
	-- coucou = { torch.rand( 3, 16, 1 ), torch.rand( 3, 2 ) }
	-- pp = self.net:forward( coucou )
	-- print( "pp" )
	-- print( pp )
	-- popopopopo()
	
	--[[
	local viewSize = self.stock_input_len * 3
	self.net:add( nn.Reshape( self.stock_input_len * 3 ) )
	self.net:add(nn.Linear(viewSize, viewSize/2))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(viewSize/2, 128))
	self.net:add(nn.ReLU())
	self.net:add(nn.Linear(128, self.number_of_actions))
	self.net:add(nn.ReLU())
	]]--
	
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
	self.stock_input_len = 24

	--- current number of iteration	
	self.iter = 0

	--- epsilon annealing
	self.ep_start = 	0.9
	self.ep	=			self.ep_start
	self.ep_end =		0.000001
	self.ep_end_t =		1000000

	self.learn =		true
	
	--- replay memory
	--- max size of replay memory
	self.replay_memory_max_size = 100000
	--- actual replay memory
	self.replay_memory = {}
	
	--- Training
	self.trainingCount = 0
	
	--- Training batch size
	self.training_batch_size = 500
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
	
	-- Target network
	self.target_net = args.target_q
    if not self.target_q then
    	self.target_net = self.agent_net
    end
    
    
end

function dqa:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    -- term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    -- term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.net
    end

    -- Compute max_a Q(s_2, a).
    q2_max = target_q_net:forward(s2):float():max(1)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.gamma)

    delta = torch.Tensor(1):fill(r):clone():float()

    --if self.rescale_r then
    --    delta:div(self.r_max)
    --end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.net:forward(s):float()
    print( "q_all=" .. tostring( q_all ) )
    
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    --if self.clip_delta then
    --    delta[delta:ge(self.clip_delta)] = self.clip_delta
    --    delta[delta:le(-self.clip_delta)] = -self.clip_delta
    --end

    local targets = torch.zeros(self.training_batch_size, self.number_of_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end

function dqa:actRandom( input )
	t = torch.Tensor(1)
	t:random(1,3)
	return t
end

function dqa:forward( input, net )
	---print("input avant: " .. tostring(input))
	--print( "input: " .. tostring(input))
	local ret
	if net ~= nil then
		ret = net:forward( input )
	else
		ret = self.net:forward( input )
	end
	return ret
end

function dqa:policy( input, net )
	local action_values = self:forward( input, net )
	local maxval = action_values[1]
  	local max_index = 1
 
 	-- find maximum output and note its index and value
  	for i = 1, self.number_of_actions do
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
	local rert = self.net:forward( input )
	print( "ret.... " .. tostring(rert) )
	print( "############# END ###########################")
	return torch.Tensor(1):fill(ret.action)
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
	
	h_inputs = torch.Tensor(self.training_batch_size, self.stock_input_len, 1 )
	p_inputs = torch.Tensor(self.training_batch_size, 2 )
	
	targets = torch.Tensor(self.training_batch_size, self.number_of_actions, 1 )
		
	print( "Training with " .. tostring( self.training_batch_size ) .. " samples" )
	
	-- Switch the target network regularly
	if self.iter % 50 == 0 then
		print( "#### Updating target network ! ####\n" )
		-- print( self.net )
		-- print( self.target_net )
		self:saveNetwork()
		-- a:saveNetwork()
		self.target_net = self.net
		-- elf.net:copy(self.target_net)
		-- self.target_net = torch.load( 'net.bin' )
	end
	
	for k = 1, self.training_batch_size do
		--- Choose tuple randomly from replay memory
		local sampleIdx = math.random( 1, table.length(self.replay_memory))
		local sample = self.replay_memory[ sampleIdx ];
		-- print( sample )
		
		---print( "test DQN #1" )
		--local targets, delta, q2_max = self:getQUpdate{s=sample[1], a=sample[2], r=sample[3], s2=sample[4], update_qmax=true}
		--print( tostring( targets ) )
		--print( "end of test")
		
		-- copy state from experience S0
        local x = sample[1];
   
   		-- compute best action for the new state S1
        local best_action = self:policy(sample[4], self.target_net);
        
        --[[ get current action output values
   				we want to make the target outputs the same as the actual outputs
   				expect for the action that was chose - we want to replace this with
	   			the reward that was obtained + the utility of the resulting state
   			--]]
   			
   		
   		local all_outputs = self.net:forward(x);
		-- inputs[k] = x:clone();      	
		h_inputs[k] = x[1]:clone()
		p_inputs[k] = x[2]:clone()
		targets[k] = all_outputs:clone();
		-- print( sample[2][1] )
		targets[k][ sample[2][1] ] = sample[3] + self.gamma * best_action.value; 
	end
	-- Concatenate all this shit
	inputs = { h_inputs, p_inputs }
	
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
	     if self.coefL1 ~= 0 or self.coefL2 ~= 0 then
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
        
        -- increment number of learning steps
		self.iter = self.iter + 1
        
end

function dqa:train( stepTuple )
	
	--- Insert tuple to memory
	self:insertToMemory( stepTuple )
	
	--- Train
	if self.learn and table.length( self.replay_memory ) > self.training_batch_size then
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
	self.ep = self.ep - 0.000001
	
	-- return choosen action
	return ret
end
