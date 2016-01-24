--[[ 
Copyright (c) 
]]--

require('torch')
require('csvigo')

local csvenv = torch.class('CSVEnvironment')

function normalizeRows( inp )

	local N = inp:size()[1]
	local C = inp:size()[2]
	
	Z = torch.sum(inp,2)
	
  	one_over_Z = torch.cdiv(torch.ones(N) , Z)
  
  	out = torch.Tensor(N, C):zero()
  
  	for k=1,N do
  		local m = inp[k]:mean()
  		local s = inp[k]:std()
  		out[k] = inp[k]
  		if s ~= 0 then 
    		out[k]:add(-m)
    		out[k]:div(s)
  		end
  	end
  	
  	return out
end

function csvenv:__init(args)
	
	--- output data format
	if args.stock_chunk_len ~= nil then
	self.stock_chunk_len = args.stock_chunk_len
	else
	self.stock_chunk_len = 16
	end

	--- CSV file we'll be using
	if args.csv_file ~= nil then
	self.csv_file = args.csv_file;
	else
	self.csv_file = "input.csv"
	end

	--- minimum interval between values in seconds
	if args.time_interval ~= nil then
	self.time_interval = args.time_interval
	else
	self.time_interval = 7200
	end

	--- current value buffer
	self.buffer = {}
	self.portfolio_buffer = {}
	self.portfolio_eur = {}
	self.portfolio_btc = {}
	
	--- Current offset in CSV file
	self.csv_offset = 1

	--- last time value encountered in CSV
	self.last_time_value = 0

	self.csv = csvigo.load({path = self.csv_file, verbose = false, mode = "raw"})
	--- print(self.csv)

	--- initial portfolio
	self.current_btc = 0
	self.current_euro = 60
	
	--- current BTC value (in euro?)
	self.current_btc_val = 0
	self.prev_btc_val = 0
	
end

function csvenv:portfolioValue()
	local btcval = self.current_btc_val * self.current_btc
	local eurval = self.current_euro
	return btcval + eurval
end

function csvenv:sell()
	if self.current_btc > 0 then
	self.current_euro = self.current_btc * self.current_btc_val
	self.current_btc = 0
	end
end

function csvenv:buy()
	if self.current_euro > 0 then
	self.current_btc = self.current_euro / self.current_btc_val
	self.current_euro = 0
	end
end

-- returns reward AND next state given action
function csvenv:act( action )
	
	print("Current BTC price: " .. tostring( self.current_btc_val ))
	
	local prevPortfolioValue = self:portfolioValue()
	
	print( "Portfolio before: " )
	print( self.current_euro )
	print( self.current_btc )
	print("\n")
	
	local action_idx = 0
	if torch.isTensor( action ) then
		action_idx = action[1]
	else
		action_idx = action
	end
	
	local reward = 0
	local impossible_move = false
	
	if action_idx == 1 then
		print("SELL")
		if( self.current_btc <= 0 ) then
			impossible_move = true
		else
			self:sell()
		end
			
	elseif action_idx == 2 then
		print("BUY")
		if( self.current_euro <= 0 ) then
			impossible_move = true
		else
			self:buy()
		end
	else
		print("DO NOTHING")
	end
	
	print( "Portfolio after: " )
	print( self.current_euro )
	print( self.current_btc )
	
	local nextState = self:getNextState()
	
	local curPortfolioValue = self:portfolioValue()
	
	print( "previous portfolio value: " .. tostring( prevPortfolioValue ))
	print( "current portfolio value: " .. tostring( curPortfolioValue ))
	
	if impossible_move ~= true then
		reward = curPortfolioValue - prevPortfolioValue
		if action_idx == 2 and reward > 0 then
			reward = reward * 2
		end
	else
		reward = -1
	end
	
	print("---------------------")
	
	return reward, nextState

end

function csvenv:getPortfolioState()
	local ret = torch.Tensor( 1, 2 );
	ret[1][1] = self.current_euro / self:portfolioValue()
	ret[1][2] = self.current_btc / self:portfolioValue()
	return ret:reshape(2)
end

function csvenv:testNewModel(x)
	
	model1 = nn.Sequential()
	model1:add(nn.TemporalConvolution(1,4,3,1))
	model1:add(nn.ReLU())
	model1:add(nn.TemporalMaxPooling(2))
	model1:add(nn.ReLU())
	local m = nn.View(-1):setNumInputDims(2)
	model1:add(m)

	model2 = nn.Sequential():add(nn.Linear(2, 1)):add(nn.Tanh())

	model3 = nn.Linear(29, 4)

	global = nn.Sequential():add(nn.ParallelTable():add(model1):add(model2)):add(nn.JoinTable(1, 1)):add(model3)
	print( "x:" )
	print( x )
	popo = global:forward( x )
	print( "popo:\n" )
	print( popo )
end

function csvenv:getNextState()
	local currow = self.csv[ self.csv_offset ]
	local timeval = tonumber(currow[ 1 ])
	while timeval - self.last_time_value < self.time_interval do
		currow = self.csv[ self.csv_offset ]
		timeval = tonumber(currow[ 1 ])
		self.csv_offset = self.csv_offset + 1
	end
	local val = tonumber(currow[2])
	
	self.prev_btc_val = self.current_btc_val
	self.current_btc_val = val
	
	--- print(timeval)
	self.last_time_value = timeval
	
	self.cur_portfolio_value = self:portfolioValue()
	
	-- append value
	if #self.buffer >= self.stock_chunk_len then
		table.remove( self.buffer, 1 )
		table.remove( self.portfolio_buffer, 1 )
		table.remove( self.portfolio_eur, 1 )
		table.remove( self.portfolio_btc, 1 )
	end 
	table.insert( self.buffer, val )
	local pfval = self:portfolioValue()
	table.insert( self.portfolio_buffer, pfval )
	table.insert( self.portfolio_eur, self.current_euro )
	table.insert( self.portfolio_btc, self.current_btc )
	
	--- print( #self.buffer ) 
	--- return torch.Tensor( {ret} ):transpose(1,2)
	if #self.buffer < self.stock_chunk_len then
		return nil
	else
		-- return torch.Tensor( {self.buffer} ):transpose(1,2)
		-- local ret = torch.Tensor( {self.buffer, self.portfolio_buffer} ):transpose(1,2)
		
		local normalizedHistory = ( torch.Tensor( {self.buffer} ) )
		normalizedHistory = normalizeRows( normalizedHistory )
		-- print( "normalizedHistory: " .. tostring( normalizedHistory ) )
		 --print( self:getPortfolioState() )
		local ret = {normalizedHistory:transpose(1,2), self:getPortfolioState() }
		-- local ret = torch.Tensor( {self.buffer, self.buffer, self.buffer} )
		-- print( "ret: \n" )
		-- print( ret )
		
		return ret
		-- print(ret:transpose(1,2))
		-- return ret:transpose(1,2)
		--print(torch.std(ret,2,true))
		--return ret
	end
end
