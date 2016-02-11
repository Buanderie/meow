
require('DeepQAgent')
require('MonkeyAgent')

require('CSVEnvironment')

require('torch')
require('gnuplot')
require('lfs')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function getTimestamp()
local tnf=os.date('%Y%m%d%H%M%S',os.time())
return tostring(tnf)
end

local netFile = "net.bin"
if not file_exists( netFile ) then
	netFile = nil
end

-- Create an agent
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

-- Create a new experiment
local experimentName = "exp_" .. getTimestamp()
local experimentDir = "./" .. experimentName
lfs.mkdir( experimentDir )
local prevLoss = a.currentLoss
--

use_plot = false

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

	if nsteps == a.learning_steps_burnin then
		csvenv.current_btc = csvenv.initial_btc
		csvenv.current_euro = csvenv.initial_euro
	end
	
	if nsteps > 20000 then	
	table.insert(value, avgReward)
	-- table.insert(time, timer:time().real)
   	table.insert( time, nsteps )
   	end
   	
   	-- save network from time to time
   	if nsteps % 10 == 0 then
   		local curLoss = a.currentLoss
   		-- print( prevLoss )
   		-- print( curLoss )
   		-- print( tostring( prevLoss - curLoss ) )
   		if prevLoss - curLoss >= 0.1 then
   			local netPath = experimentDir .. "/net_" .. tostring(curLoss) .. ".net"
   			print( netPath )
   			a:saveNetwork( netPath )
   			prevLoss = curLoss
   		end
   	end
   	
	-- plot reward
	if use_plot then
		if nsteps % 10 == 0 and nsteps > 20000 then
		cgtime = torch.Tensor(time)
		cgevaluations = torch.Tensor(value)
		gnuplot.figure(1)
		gnuplot.title('Average reward over time')
		gnuplot.plot(cgtime, cgevaluations)
		end
	end
	
	nsteps = nsteps + 1
	
end
