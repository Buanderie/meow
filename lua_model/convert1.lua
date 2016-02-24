

require('csvigo')

outcsv = csvigo.File("out.csv", "w")

csv = csvigo.load({path = arg[1], verbose = false, mode = "raw"})
local curidx = #csv
local currow = csv[ curidx ]
local curtime = 0
while currow ~= nil do
	local curvalue = currow[8]
	local curvolume = currow[7]
	local curins = { curtime, curvalue, curvolume }
	outcsv:write( curins )
	curidx = curidx - 1
	currow = csv[ curidx ]
	curtime = curtime + 86400
end
