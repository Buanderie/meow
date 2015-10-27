
require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'
require 'math'

--[[
-- Read CSV file
-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end
local filePath = 'krakenEUR.csv'
-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end
local ROWS = i - 1  -- Minus 1 because of header
-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()
local data = torch.Tensor(ROWS, 1)
local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
  	if key == 2 then
    	data[i][1] = val
  	end
  end
  -- print(data[i][2])
end
csvFile:close()
-- Serialize tensor
local outputFilePath = 'train.th7'
torch.save(outputFilePath, data)
]]--

-- threads
torch.setnumthreads(7)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

dataset = {}
test_dataset = {}

local data = torch.load('/home/said/train.th7')
data = data:transpose(1,2)
   
local n_samples = (data:size()[2] - 512)/2
n_samples = n_samples / 2000
local sample_size = 256
local target_size = sample_size / 4
-- First half is for training
local shuffle = torch.randperm(data:size()[2] / 2) -- shuffle the data
local input = torch.FloatTensor(n_samples, sample_size)
local target = torch.FloatTensor(n_samples, target_size)
function dataset:size() return n_samples end
   
-- Training dataset
print(data:size())
for i=1,n_samples do
 	local idx = shuffle[i]
 	local wholei = (data:sub(1, 1, idx, idx+sample_size+target_size-1):transpose(1,2))
 	local min = torch.min(wholei)
 	local max = torch.max(wholei)
 	local maxmin = (max - min)
 	local traini = (data:sub(1, 1, idx, idx+(sample_size-1)):transpose(1,2))
 	local testi  = (data:sub(1, 1, idx+sample_size,idx+sample_size+target_size-1):transpose(1,2))
 	
 	dataset[i] = { (traini - min)/maxmin, (testi - min)/maxmin }
 	--print( dataset[idx][1] )
 	-- input[idx] = data:sub(1, 1, i, i+(sample_size-1))
 	-- target[idx] = data:sub(1, 1, i+sample_size,i+sample_size+target_size-1)
end

local shuffle = torch.randperm(data:size()[2] / 2) -- shuffle the data
-- Testing dataset
for i=1,n_samples do
 	local idx = shuffle[i] + (data:size()[2] / 2)
 	local wholei = (data:sub(1, 1, idx, idx+sample_size+target_size-1):transpose(1,2))
 	local min = torch.min(wholei)
 	local max = torch.max(wholei)
 	local maxmin = (max - min)
 	local traini = (data:sub(1, 1, idx, idx+(sample_size-1)):transpose(1,2))
 	local testi  = (data:sub(1, 1, idx+sample_size,idx+sample_size+target_size-1):transpose(1,2))
 	
 	test_dataset[i] = { (traini - min)/maxmin, (testi - min)/maxmin }
 	--print( dataset[idx][1] )
 	-- input[idx] = data:sub(1, 1, i, i+(sample_size-1))
 	-- target[idx] = data:sub(1, 1, i+sample_size,i+sample_size+target_size-1)
end

function test_model()
local shuffle = torch.randperm(n_samples) -- shuffle the data
for i = 1,20 do
	idx = shuffle[i]
   time = {}
   local myPrediction = net:forward(test_dataset[idx][1])
   -- print( myPrediction )
   -- print( dataset[i][2] )
   -- print( "-----------------------------------" )
   for j=1,sample_size+target_size do
   	table.insert(time, j)
   end
   popo = torch.Tensor(time)
   pipi = torch.cat(test_dataset[idx][1],test_dataset[idx][2],1)
   pupu = torch.cat(test_dataset[idx][1],myPrediction,1)
   	gnuplot.pngfigure('/tmp/plot_' .. i .. '.png')
		gnuplot.title('CG loss minimisation over time')
		gnuplot.plot({'pred', popo, pupu}, {'data', popo, pipi})
		gnuplot.plotflush()
end
end

print( dataset[1][1]:size() )

net = nn.Sequential()
--net:add(nn.AddConstant(0.00000001))
net:add(nn.TemporalConvolution(1,16,5,1)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(2))
net:add(nn.TemporalConvolution(16,32,5,1)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(2))
net:add(nn.View(1952)) 
net:add(nn.ReLU())
net:add(nn.Linear(1952, 1952/2))
net:add(nn.Tanh())
net:add(nn.Linear(1952/2, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128, target_size))
--net:add(nn.LogSoftMax())

-- a = torch.randn(sample_size,1)
-- ptpt = net:forward( dataset[1][1] )
-- print(ptpt:size())
-- exit()

a = torch.randn(3,1)
b = torch.randn(3,1)
c = torch.randn(3,1)

d = torch.cat(a,b,1)

print(d)
----------------------------------------------------------------------
-- 3. Define a loss function, to be minimized.

-- In that example, we minimize the Mean Square Error (MSE) between
-- the predictions of our linear model and the groundtruth available
-- in the dataset.

-- Torch provides many common criterions to train neural networks.

criterion = nn.MSECriterion()


----------------------------------------------------------------------
-- 4. Train the model

-- To minimize the loss defined above, using the linear model defined
-- in 'model', we follow a stochastic gradient descent procedure (SGD).

-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the 
-- entire training set is too costly.

-- Given an arbitrarily complex model, we can retrieve its trainable
-- parameters, and the gradients of our loss function wrt these 
-- parameters by doing so:

x, dl_dx = net:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our model, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#dataset) then _nidx_ = 1 end

   local sample = dataset[_nidx_]
   local target = sample[2]      -- this funny looking syntax allows
   local inputs = sample[1]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accomodate 
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(net:forward(inputs), target)
   net:backward(inputs, criterion:backward(net.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- Timing training
-- reset the timer!
timer = torch.Timer()
timer:reset()
time = {}
loss = {}
-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

-- we cycle 1e4 times over our training data
for i = 1,9e2 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,n_samples do

			-- print( "sample " .. i .. "/" .. n_samples .. "\n" )
      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

		print( current_loss .. " / " .. n_samples )
   -- report average error on epoch
   current_loss = current_loss / (n_samples)
   print('current loss = ' .. i .. " - " .. current_loss)
   
   if math.fmod( i, 10 ) == 0 then
   --[[
   table.insert(loss, current_loss)
   table.insert(time, timer:time().real)

		sgdtime = torch.Tensor(time)
		sgdloss = torch.Tensor(loss)
		gnuplot.figure(1)
		gnuplot.title('SGD loss minimisation over time')
		gnuplot.plot(sgdtime, sgdloss)
		]]--
		test_model()
	end
	
end

----------------------------------------------------------------------
-- 5. Test the trained model.

-- Now that the model is trained, one can test it by evaluating it
-- on new samples.

-- The text solves the model exactly using matrix techniques and determines
-- that 
--   corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

-- We compare our approximate results with the text's results.


