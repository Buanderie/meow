
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
losses = {}
totReward = 0
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

	timer = torch.Timer()
	local action = a:actOnInput( state )
	local t = timer:time().real
	print( "AGENT_actOnInput_t=" .. tostring(t) )
	
	timer = torch.Timer()
	local reward, nextState = csvenv:act( action )
	local t = timer:time().real
	print( "ENV_act_t=" .. tostring(t) )
	
	timer = torch.Timer()
	local stepTuple = { state, action, reward, nextState }
	a:train( stepTuple )
	local t = timer:time().real
	print( "AGENT_train_t=" .. tostring(t) )
	
	-- prepare next step
	state = nextState
	
	print( "Reward: " .. tostring(reward ) )
	nsteps = nsteps + 1
	totReward = totReward + reward
	avgReward = totReward / nsteps
	
	--if #value >= 100 then
	--	table.remove( value, 1 )
	--	table.remove( time, 1 )
	--end 

	if nsteps == a.learning_steps_burnin then
		csvenv.current_btc = csvenv.initial_btc
		csvenv.current_euro = csvenv.initial_euro
	end
	
	if nsteps > a.learning_steps_burnin then	
	table.insert(value, avgReward)
	table.insert( losses, a.currentLoss)
   	table.insert( time, nsteps )
   	end
   	
   	if a.currentLoss > 10 then
   	print("#####################SHIT#######################")
   	end
   	
   	-- save network from time to time
   	if nsteps % 10 == 0 then
   		local curLoss = a.currentLoss
   		-- print( prevLoss )
   		-- print( curLoss )
   		-- print( tostring( prevLoss - curLoss ) )
   		if prevLoss - curLoss >= 0.01 then
   			local netPath = experimentDir .. "/net_" .. tostring(curLoss) .. ".net"
   			print( netPath )
   			a:saveNetwork( netPath )
   			prevLoss = curLoss
   		end
   	end
   	
	-- plot reward
	if use_plot then
		if nsteps % 500 == 0 and nsteps > a.learning_steps_burnin then
		cgtime = torch.Tensor(time)
		cgevaluations = torch.Tensor(value)
		gnuplot.figure(1)
		gnuplot.title('Average reward over time')
		gnuplot.plot(cgtime, cgevaluations)
		gnuplot.figure(2)
		gnuplot.title('loss over time')
		gnuplot.plot(cgtime, torch.Tensor(losses))
		end
	end
	
	-- release emory from time to time
	if nsteps % 2000 == 0 then
		collectgarbage()
	end
	
	nsteps = nsteps + 1
	
end
