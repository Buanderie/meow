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
 
	--- Current offset in CSV file
	self.csv_offset = 1

	--- last time value encountered in CSV
	self.last_time_value = 0

	self.csv = csvigo.load({path = self.csv_file, verbose = false, mode = "raw"})
	--- print(self.csv)

end

function csvenv:getNextState()
	local ret = {}
	local i = 0
	local offset = self.csv_offset
	while i < self.stock_chunk_len do
		local currow = self.csv[ offset ]
		local timeval = tonumber(currow[ 1 ])
		if timeval - self.last_time_value > self.time_interval then
			if i == 0 then self.csv_offset = offset end
			--- print( timeval )
			--- print( self.last_time_value )
			local curask = tonumber(currow[ 2 ])
			table.insert( ret, curask )
			i = i + 1	
			self.last_time_value = timeval
		end	
		offset = offset + 1
	end
	return torch.Tensor( {ret} ):transpose(1,2)
end
