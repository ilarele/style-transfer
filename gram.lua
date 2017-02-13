require 'nn'

if gpu then
   require 'cutorch'
   require 'cunn'
end


local Gram = torch.class('nn.Gram', 'nn.Module')
--torch.Tensor = torch.CudaTensor
function Gram:updateOutput(input)
   gpu = false
   if input:type() == 'torch.CudaTensor' then
      gpu = true
      --torch.Tensor = torch.CudaTensor
   end

   local inp = input:clone()
   size 	= inp:size()
   inp:resize(size[1],size[2], size[3] * size[4])
   if gpu == true then
      self.output = torch.zeros(size[1],size[2], size[2]):cuda()
   else 
      self.output = torch.zeros(size[1],size[2], size[2])
   end
   -- pentru fiecare element din batch aflam o matrice gram
   for i = 1, size[1] do
      aux = inp[i]
      self.output[i] = aux * aux:t()
   end

  return self.output / (size[2] * size[3] * size[4]);
end

function Gram:updateGradInput(input, gradOutput)
   gpu = false
   if input:type() == 'torch.CudaTensor' then
      gpu = true
      --torch.Tensor = torch.CudaTensor
   end
   
   local inp = input:clone()
   size  = inp:size()
   
   if gpu == true then
      self.gradInput = torch.zeros(size[1],size[2], size[3],size[4]):cuda()
      input2col = torch.zeros(size[1], size[3] * size[4], size[2]):cuda()
   else 
      self.gradInput = torch.zeros(size[1],size[2], size[3],size[4])
      input2col = torch.zeros(size[1], size[3] * size[4], size[2])
   end

   -- rearanjam tensorul de intrare de dim BxCxHxW intr-un tensor BxHWxC
   for b = 1, size[1] do
      local linie = 0
      for h = 1, size[3] do
         for w = 1, size[4] do
            linie = linie + 1
            input2col[{ b,linie,{} }] = inp[{ b, {}, h,w}]              
         end
      end
   end
   -- singurul calcul din functie
   for i = 1, size[1] do
      input2col[i] = input2col[i] * gradOutput[i] * 2;

   end

   -- rearanjam tensorul input2col de dim BxHWxC intr-un tensor  BxCxHxW
   for b = 1, size[1] do
      linie = 0;
      for h = 1, size[3] do
         for w = 1, size[4] do
            linie = linie + 1
            self.gradInput[{ b, {}, h,w}] = input2col[{ b,linie,{} }]              
         end
      end
   end

   return self.gradInput / (size[2] * size[3] * size[4])
end
