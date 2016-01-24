
require('DeepQAgent')
require('MonkeyAgent')

require('CSVEnvironment')

require('torch')
require('gnuplot')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local netFile = "net.bin"
if not file_exists( netFile ) then
	netFile = nil
end

-- Create an agent
-- local a = DeepQAgent{}
local a = DeepQAgent{agent_net=netFile}

-- Create an environment
local csvenv = CSVEnvironment{csv_file="./krakenEUR.csv"}

for i=1,257 do
ret2 = csvenv:getNextState()
-- print(ret2)
end

-- test = torch.randn( a.stock_input_len, 1 );
--- print(test)

timer = torch.Timer()
time = {}
value = {}
avgReward = 0
nsteps = 0

-- initial state
local state = csvenv:getNextState()
for i=1,10000 do
	-- coucou = torch.Tensor(1)
	-- print( coucou[1] )
	-- ret2 = csvenv:getNextState()
	-- print(ret2:size())
	---ret = a:chooseAction(csvenv:getNextState())
	-- print(ret)
	-- print(a:actOnInput( csvenv:getNextState() ))
	local action = a:actOnInput( state )
	local reward, nextState = csvenv:act( action )
	
	local stepTuple = { state, action, reward, nextState }
	a:train( stepTuple )
	
	-- prepare next step
	state = nextState
	
	print( "Reward: " .. tostring(reward ) )
	nsteps = nsteps + 1
	avgReward = avgReward + ( reward - avgReward ) / nsteps
	table.insert(value, avgReward)
	table.insert(time, timer:time().real)
   
	-- plot reward
	if i % 10 == 0 then
	cgtime = torch.Tensor(time)
	cgevaluations = torch.Tensor(value)
	--gnuplot.figure(1)
	--gnuplot.title('Average reward over time')
	--gnuplot.plot(cgtime, cgevaluations)
	a:saveNetwork()
	end
	
end
