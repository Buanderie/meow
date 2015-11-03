
require('DeepQAgent')
require('CSVEnvironment')
require('torch')

-- Create an agent
local a = DeepQAgent( {popo=2} )

-- Create an environment
local csvenv = CSVEnvironment{csv_file="./krakenEUR.csv"}

ret2 = csvenv:getNextState()
--- print(ret2:transpose(1,2))

test = torch.randn( a.stock_input_len, 1 );
--- print(test)

for i=1,10 do
ret = a:chooseAction(ret2)
print(ret)
end
