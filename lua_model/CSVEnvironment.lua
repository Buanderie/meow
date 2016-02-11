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
	self.stock_chunk_len = 24
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
	self.time_interval = 3600
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
	self.initial_btc = 0.11
	self.initial_euro = 0
	
	-- current protfolio
	self.current_btc = 0
	self.current_euro = 0
	
	--- other portfolio info
	self.value_at_entry = 0
	
	--- current timestamp we're in
	self.current_timestamp = 0
	
	--- current BTC value (in euro?)
	self.current_btc_val = 0
	self.prev_btc_val = 0
	self.value_at_entry = 0.0000001
	
	--- should we use reporting ?
	self.use_reporting = true --args.use_reporting
	if self.use_reporting then
		self.report_csv = csvigo.File("report.csv", "w")
		self.report_csv:write( {"timestamp", "current_btc_price", "action_taken"} )
	end
	
	--- GPU ?
	self.gpu = false
	
	-- Init
	self:reset()
	
end

function csvenv:portfolioValue()
	local btcval = self.current_btc_val * self.current_btc
	local eurval = self.current_euro
	return btcval + eurval
end

function csvenv:sell()
	if self.current_btc > 0 then
		self.current_euro = (self.current_btc * self.current_btc_val)
		self.current_euro = self.current_euro - self:getFees( self.current_euro )
		self.current_btc = 0
		self.value_at_entry = self.current_btc_val
	end
end

function csvenv:buy()
	if self.current_euro > 0 then
		self.current_btc = self.current_euro / self.current_btc_val
		self.current_btc = self.current_btc - self:getFees( self.current_btc )
		self.current_euro = 0
		self.value_at_entry = self.current_btc_val
	end
end

function csvenv:reportAction( action )
	self.report_csv:write( { self.current_timestamp, self.current_btc_val, action } )
end

function csvenv:getFees( value )
	print( "fees: " .. tostring( (0.3/100) * value ) )
	return (0.03/100) * value
end

-- returns reward AND next state given action
function csvenv:act( action )
	
	local coucou = ((math.max(self.value_at_entry, 0.0000001) -  self.current_btc_val) /  self.current_btc_val) / 100
	print("coucou: " .. tostring(coucou ) ) 
	print("Previous BTC price: " .. tostring( self.current_btc_val ))
	
	local vEntry = self.value_at_entry
	local prevRet = (self.current_btc_val - vEntry) / vEntry
	print( "Previous return: " .. tostring(prevRet) )
	
	local prevPortfolioValue = self:portfolioValue()
	local prevBtcVal = self.current_btc_val
	
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
	
	local curRet = (self.current_btc_val - vEntry) / vEntry
	print( "Current BTC price: " .. tostring( self.current_btc_val ))
	print( "Current return: " .. tostring(curRet) )
	
	local curPortfolioValue = self:portfolioValue()
	local curBtcVal = self.current_btc_val
	
	print( "previous portfolio value: " .. tostring( prevPortfolioValue ))
	print( "current portfolio value: " .. tostring( curPortfolioValue ))
	
	local pfReturn = (curPortfolioValue - prevPortfolioValue) / prevPortfolioValue
	-- local pfReturn = curRet - prevRet
	local btcReturn = (curBtcVal - prevBtcVal) / prevBtcVal
	
	--[[
	print( "pfReturn: " .. tostring(pfReturn) )
	print( "btcReturn: " .. tostring(btcReturn) )
	
	--if impossible_move == true then
	--	reward = 0
	--else
		-- reward = pfReturn
		
		if btcReturn > 0 then
			if pfReturn <= 0 then
				reward = -btcReturn
			else
				reward = pfReturn
			end
		else
			if pfReturn >= 0 then
				reward = -btcReturn
			else
				reward = pfReturn
			end
		end
		
	if action_idx ~= 0 then
		reward = 10 * reward
	end
	]]--
	
	if btcReturn > 0 then
	    if pfReturn > 0 then
	        reward = 5 * math.abs(pfReturn)
	    else
	        reward = -1 * math.abs(btcReturn)
	    end
	else
	    if pfReturn >= 0 then
	        reward = 1 * math.abs(btcReturn)
	    else
	        reward = -5 * math.abs(pfReturn);
	    end
	end

	--end
		-- if action_idx == 2 and reward > 0 then
		--	reward = reward * 2
		-- end
	--else
	--	reward = -1
	--end
	
	-- EXP
	-- reward = reward * 1000
	--
	
	-- Penalty for doing shit
	if impossible_move then
		reward = reward - 0.1
	end
	--
	
	print("REWARD: " .. tostring( reward ) )
	print("---------------------")
	
	-- reporting
	if self.use_reporting then
		if impossible_move == false then
			self:reportAction( action_idx )
		end
	end
	
	--
	
	return reward * 2, nextState

end

function csvenv:getPortfolioState()
	local ret = torch.Tensor( 1, 3 );
	ret[1][1] = self.current_euro / self:portfolioValue()
	ret[1][2] = self.current_btc / self:portfolioValue()
	ret[1][3] = ((math.max(self.value_at_entry, 0.0000001) -  self.current_btc_val) /  self.current_btc_val) / 10
	return ret:reshape(3)
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

function csvenv:reset()
	self.csv_offset = 1
	self.current_timestamp = 0
	self.current_btc_val = 0
	self.prev_btc_val = 0
	self.last_time_value = 0
	self.current_btc = self.initial_btc
	self.current_euro = self.initial_euro
	
	for i,v in ipairs(self.buffer) do table.remove(self.buffer, i) end
	popo = nil
	while popo == nil do
		popo = self:getNextValue()
	end
end

function csvenv:getNextValue()

	local currow = self.csv[ self.csv_offset ]
	
	if currow == nil then
		-- print("nil1 " .. tostring(self.csv_offset) )
		self:reset()
		currow = self.csv[ self.csv_offset ]
	end
	
	local timeval = tonumber(currow[ 1 ])
	while timeval - self.last_time_value < self.time_interval do
		currow = self.csv[ self.csv_offset ]
		
		if currow == nil then
			-- print("nil2")
			self:reset()
			currow = self.csv[ self.csv_offset ]
		end
		
		timeval = tonumber(currow[ 1 ])
		self.csv_offset = self.csv_offset + 1
		
	end
	
	self.last_time_value = timeval
	local val = tonumber(currow[2])
	
	self.current_timestamp = timeval
	self.prev_btc_val = self.current_btc_val
	self.current_btc_val = val
	
	return val
	
end

function csvenv:getNextState()

	while #self.buffer < self.stock_chunk_len do
		local val = self:getNextValue()
		table.insert( self.buffer, val )
	end
	
	local normalizedHistory = ( torch.Tensor( {self.buffer} ) )
	normalizedHistory = normalizeRows( normalizedHistory )

	local ret = nil
	if self.gpu then
		ret = {normalizedHistory:transpose(1,2):cuda(), self:getPortfolioState():cuda() }
	else
		ret = {normalizedHistory:transpose(1,2), self:getPortfolioState() }
	end
	
	table.remove( self.buffer, 1 )
	
	return ret

end
