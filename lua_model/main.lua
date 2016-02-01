
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

timer = torch.Timer()
time = {}
value = {}
avgReward = 0
nsteps = 0

-- initial state
local state = csvenv:getNextState()
print("init_state " .. tostring(state))
local nsteps = 1

-- test
	-- a:actThompson( state )
	-- exit(0)
--

while true do

	local action = a:actOnInput( state )
	local reward, nextState = csvenv:act( action )
	
	local stepTuple = { state, action, reward, nextState }
	a:train( stepTuple )
	
	-- prepare next step
	state = nextState
	
	print( "Reward: " .. tostring(reward ) )
	nsteps = nsteps + 1
	avgReward = avgReward + ( reward - avgReward ) / nsteps
	
	--[[
	if #value >= 100 then
		table.remove( value, 1 )
		table.remove( time, 1 )
	end 
	]]--

	if nsteps == 20000 then
		csvenv.current_btc = 0
		csvenv.current_euro = csvenv.initial_euro
	end
	
	if nsteps > 20000 then	
	table.insert(value, avgReward)
	-- table.insert(time, timer:time().real)
   	table.insert( time, nsteps )
   	end
   	
	-- plot reward
	if nsteps % 10 == 0 and nsteps > 20000 then
	cgtime = torch.Tensor(time)
	cgevaluations = torch.Tensor(value)
	gnuplot.figure(1)
	gnuplot.title('Average reward over time')
	gnuplot.plot(cgtime, cgevaluations)
	
	end
	
	nsteps = nsteps + 1
	
end
