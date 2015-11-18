--[[ 
Copyright (c) 
]]--

require('torch')
require('csvigo')

local csvenv = torch.class('CSVEnvironment')

function csvenv:__init(args)
	
	--- output data format
	if args.stock_chunk_len ~= nil then
	self.stock_chunk_len = args.stock_chunk_len
	else
	self.stock_chunk_len = 256
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
	self.time_interval = 240
	end

	--- current value buffer
	self.buffer = {}
	
	--- Current offset in CSV file
	self.csv_offset = 1

	--- last time value encountered in CSV
	self.last_time_value = 0

	self.csv = csvigo.load({path = self.csv_file, verbose = false, mode = "raw"})
	--- print(self.csv)

	--- initial portfolio
	self.current_btc = 60
	self.current_euro = 0

	--- current BTC value (in euro?)
	self.current_btc_val = 0

end

function csvenv:sell()

end

function csvenv:buy()

end

function csvenv:act( action_idx )
	
	local reward = 0

	if action_idx == 1 then
		-- selling
			
	elseif action_idx == 2 then
		-- buying 
		
	end

	return reward

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
	self.current_btc_val = val
	--- print(timeval)
	self.last_time_value = timeval
	-- append value
	if #self.buffer >= self.stock_chunk_len then
		table.remove( self.buffer, 1 )
	end 
	table.insert( self.buffer, val )
	--- print( #self.buffer ) 
	--- return torch.Tensor( {ret} ):transpose(1,2)
	if #self.buffer < self.stock_chunk_len then
		return nil
	else
		return torch.Tensor( {self.buffer} ):transpose(1,2)
	end
end
