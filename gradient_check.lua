--------------------------------------------------------------------------------
--- Imports
--------------------------------------------------------------------------------

require "torch"
require "nn"
require 'gram'
--require 'cutorch'
--require 'cunn'


--------------------------------------------------------------------------------
--- Do gradient check for a model with no params,
--- According to a criterion and a given test case
--------------------------------------------------------------------------------

function inputGradientCheck(model, criterion, inputs, outputs, epsilon)
   if inputs:type() == 'torch.CudaTensor' then
      gpu = true
      torch.Tensor = torch.CudaTensor
   end

   size = inputs:size()
   -- Computed gradient
   criterion:forward(model:forward(inputs), outputs)                  -- forward
   derr = criterion:backward(model.output, outputs)
   origGradient = model:backward(inputs, derr)  --backward


   --- Definition gradient 
   local defGradient = torch.zeros(size) 
   for b = 1, size[1] do
      for ch = 1, size[2] do
         for h = 1, size[3] do
            for w = 1, size[4] do
               i1 = inputs:clone()
               i1[b][ch][h][w] = i1[b][ch][h][w] + epsilon
               j1 = criterion:forward(model:forward(i1), outputs)

               
               i1[b][ch][h][w] = i1[b][ch][h][w] - 2 * epsilon
               j2 = criterion:forward(model:forward(i1), outputs)
               defGradient[b][ch][h][w] = (j1 - j2) / (2 * epsilon)
            end
         end
      end
   end

   local distance = torch.norm(defGradient - origGradient)
   
   if distance < 5 * 1e-10 then
      print("Gradients are similar enough!")
   else
      print("Gradients are different!")
   end
   print("Average Grad Distance: " .. distance)
   
end

--------------------------------------------------------------------------------
--- Test gradients 
--------------------------------------------------------------------------------
gpu = false
-- todo: the gradient numerical computations on the gpu yields different results 
-- probably due to numerical errors(gpu uses float numbers)
N = 10
local model = nn.Sequential()
model:add(nn.Gram())

local criterion = nn.MSECriterion()                        -- choose a criterion


local testInputs = torch.rand(N,3,20,20)
testInputs:mul(0.001)


local testOutputs   = torch.rand(N,3,3)
-- the gram matrix is symetric
-- the matrix representing output gradients must be symetric 
for i = 1,N do
   testOutputs[{i,{},{}}] = testOutputs[{i,{},{}}] * testOutputs[{i,{},{}}]:t()
end 

local epsilon = 1e-3                       -- noise introduced in each parameter

if gpu == true then
    model       = model:cuda()
    criterion   = criterion:cuda()
    testOutputs = testOutputs:cuda()
    testInputs  = testInputs:cuda()
end
--------------------------------------------------------------------------------
--- Actual testing
--------------------------------------------------------------------------------

inputGradientCheck(model, criterion, testInputs, testOutputs, epsilon)
