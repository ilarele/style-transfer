require "torch"
require "nn"
require "rnn"
require "nngraph"


function createGram(C,H,W)

   local Xi = nn.Identity()()

   local restb = {}
   for i=1,C do
      for j=1,C do
         local xi = nn.Select(2,i)(Xi)
         local yj = nn.Select(2,j)(Xi) -- symmetric
         local gij = nn.Sum(2)(nn.Sum(3)(nn.CMulTable()({xi, yj})))
         restb[#restb + 1] = nn.Reshape(1,1,true)(gij)
      end
   end

   res = nn.JoinTable(2)(restb)
   res_reshaped = nn.Reshape(C,C,true)(res)

   return nn.gModule({Xi, Yi}, {res_reshaped})--_reshaped}) 

end

