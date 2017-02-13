--------------------------------------------------------------------------------
-- Initial setup script
-- Implies train.lua mai running point, as it employes global variables
--------------------------------------------------------------------------------

gpu = tonumber(params.gpu)
-- gpu specific code
if gpu ~= -1 then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(gpu)
else
   print("gpu id is " .. tostring(gpu))
end
