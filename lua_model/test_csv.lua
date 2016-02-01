
require('DeepQAgent')
require('MonkeyAgent')

require('CSVEnvironment')

require('torch')
require('gnuplot')

-- Create an environment
local csvenv = CSVEnvironment{csv_file="./krakenEUR2.csv"}
print("coucou")

for i=1,257 do
	print( "##### i=" .. tostring(i) .. " #####" )
	ret = csvenv:getNextValue()
	print( ret )
	print ("##########\n")
end

for i=1,257 do
	print( "##### i=" .. tostring(i) .. " #####" )
	ret = csvenv:getNextState()[ 1 ]
	print( ret )
	print ("##########\n")
end
