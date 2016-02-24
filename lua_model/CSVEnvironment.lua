--[[ 
Copyright (c) 
]]--

require('cutorch')

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
	self.initial_btc = 0
	self.initial_euro = 1000
	self.initial_portfolio_value = 0
	
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
	
	--- other analytics
	self.returnHistory = {}
	self.returnHistoryLength = 5
	
	--- should we use reporting ?
	self.use_reporting = true --args.use_reporting
	if self.use_reporting then
		self.report_csv = csvigo.File("report.csv", "w")
		self.report_csv:write( {"timestamp", "portfolio_value", "current_btc_price", "action_taken"} )
	end
	
	--- GPU ?
	self.gpu = false
	
	-- Init
	self:reset()
	
end

function csvenv:addReturn( val )
	if #self.returnHistory >= self.returnHistoryLength then
		table.remove( self.returnHistory, 1 )
	end
	table.insert( self.returnHistory, val )
end

function csvenv:getSharpeRatio()
	if #self.returnHistory < self.returnHistoryLength then
		return 0
	else
		local m = 0
		-- compute mean
		for i = 1, #self.returnHistory do
			m = m + self.returnHistory[ i ]
		end
		m = m / #self.returnHistory
		
		local v = 0
		-- compute variance
		for i = 1, #self.returnHistory do
			v = v + (self.returnHistory[i] - m)*(self.returnHistory[i] - m)
		end
		v = v / #self.returnHistory
		
		if v == 0 then
			return 0
		else
			return (m / math.sqrt(v))
		end
	end
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
		-- self.value_at_entry = self.current_btc_val
	end
end

function csvenv:buy()
	if self.current_euro > 0 then
		self.current_btc = self.current_euro / self.current_btc_val
		self.current_btc = self.current_btc - self:getFees( self.current_btc )
		self.current_euro = 0
		self.value_at_entry = self.current_btc_val * self.current_btc
	end
end

function csvenv:reportAction( action )
	self.report_csv:write( { self.current_timestamp, self:portfolioValue(), self.current_btc_val, action } )
end

function csvenv:getFees( value )
	print( "fees: " .. tostring( (0.3/100) * value ) )
	return (0.3/100) * value
end

-- returns reward AND next state given action
function csvenv:act( action )
	
	local coucou = ((math.max(self.value_at_entry, 0.0000001) -  self.current_btc_val) /  self.current_btc_val) / 100
	print("coucou: " .. tostring(coucou ) ) 
	print("Previous BTC price: " .. tostring( self.current_btc_val ))
	
	local prevPortfolioValue = self:portfolioValue()
	local prevBtcVal = self.current_btc_val
	
	local prevValue = self.value_at_entry
	print("prevValue: " .. tostring(prevValue))

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
	elseif action_idx == 3 then
		print("DO NOTHING")
	end
			
	print("curValue: " .. tostring(self.current_euro))
	print("prevValue: " .. tostring(self.value_at_entry))
	
	local curValue =  self.current_euro
	-- local curValue = self.initial_portfolio_value
	
	print("curValue: " .. tostring(curValue))
	
	local curRet = (curValue - prevValue)/prevValue
		
	print( "Portfolio after: " )
	print( self.current_euro )
	print( self.current_btc )
	
	local nextState = self:getNextState()
	
	print( "Current BTC price: " .. tostring( self.current_btc_val ))

	local curPortfolioValue = self:portfolioValue()
	local curBtcVal = self.current_btc_val
	
	print( "previous portfolio value: " .. tostring( prevPortfolioValue ))
	print( "current portfolio value: " .. tostring( curPortfolioValue ))
	
	local pfReturn = (curPortfolioValue - prevPortfolioValue) / prevPortfolioValue
	self:addReturn( pfReturn )
		
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
	
	local sr = self:getSharpeRatio()
	print( "Sharpe Ratio: " .. tostring(sr) )
	
	reward = 0

	local compVal = pfReturn
	
	--[[
	if action_idx == 1  and not impossible_move then
		if compVal > 0 then
			reward = 2
		elseif compVal < 0 then
			reward = -2
		else
			reward = 0
		end
	else
	if compVal > 0 then
			reward = 1
		elseif compVal < 0 then
			reward = -1
		else
			reward = 0
		end
	end
	]]--
	
	
		if compVal > 0 then
			reward = 1
		elseif compVal < 0 then
			reward = -1
		else
			reward = 0
		end
		-- reward = pfReturn
		
		--if reward < 0 then
		--	reward = reward * 100
		--end
		
		--[[
		if action_idx == 1  and not impossible_move then
			if curRet > 0 then
				reward = reward + 2
			elseif curRet < 0 then
				reward = reward - 4
			end
		end
		]]--
		
		-- reward = reward / 2
	
	--[[
	if btcReturn > 0 then
	    if pfReturn > 0 then
	        reward = pfReturn
	    else
	        reward = -pfReturn
	    end
	else
	    if pfReturn >= 0 then
	        reward = -btcReturn
	    else
	        reward = pfReturn
	    end
	end
	]]--
	
	--[[
	if action_idx == 1  and not impossible_move then
		if compVal > 0 then
			reward = 100 * compVal
		elseif compVal < 0 then
			reward = -100 * compVal
		else
			reward = 0
		end
	end
	]]--
	
	--[[
	if reward < 0 then
		reward = math.max( reward, -1 )
	elseif reward > 0 then
		reward = math.min( reward, 1 )
	else
		reward = 0
	end
	]]--
	
	--end
	
	--[[
	if btcReturn > 0 then
	    if pfReturn > 0 then
	        reward = 1
	    else
	        reward = -1
	    end
	else
	    if pfReturn >= 0 then
	        reward = 1
	    else
	        reward = -1
	    end
	end
	]]--
	
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
	 	-- reward = reward - 0.1
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
	
	return reward, nextState

end

function csvenv:getPortfolioState()
	-- if( self.current_euro <= 0 and self.cu
	local ret = torch.Tensor( 1, 3 );
	ret[1][1] = self.current_euro / self:portfolioValue()
	ret[1][2] = (self.current_btc * self.current_btc_val) / self:portfolioValue()
	local curReturn = 0
	--if self.current_euro > 0 then
		curReturn = ((self.current_euro - (self.value_at_entry)) /  self.value_at_entry)
	--else
	---	curReturn = 1
	--end
	ret[1][3] = curReturn
	-- ret[1][3] = 0
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
	-- self.initial_portfolio_value = self:portfolioValue()
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

	local prevValue = 0
	while #self.buffer < self.stock_chunk_len do
		local val = self:getNextValue()
		local pf  = self:portfolioValue()
		local rval = 0
		if val > 0 then
			rval = (val-prevValue)/val
		end
		table.insert( self.buffer, val )
		table.insert( self.portfolio_buffer, pf )
		prevValue = val
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
	table.remove( self.portfolio_buffer, 1 )
	
	return ret

end
