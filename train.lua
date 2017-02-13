require 'loadcaffe'
require 'image'
require 'nn'
require "optim"
require "gram"

require "cmd_options"
require "setup"
require "vgg_new"

if params.show_images == true then
   require 'qtwidget'
end


function getCriterion(opts,n)
   if opts.train_type == 'net-content' or opts.train_type == 'net-style'  then
      return nn.MSECriterion()
   elseif opts.train_type == 'netcombined' then
      local mse1 = nn.MSECriterion()
      local parallelCriterion = nn.ParallelCriterion():
                  add(mse1, opts.rv1)
      for i = 1,n do
         local mse2 = nn.MSECriterion()
         parallelCriterion:add(mse2, opts.rv2 / n)
      end
      return parallelCriterion -- all this is now redundant
   elseif opts.train_type == "lbfgs" or opts.train_type == "autoencoder" then
      local mse1 = nn.MSECriterion()
      local parallelCriterion = nn.ParallelCriterion():
         add(mse1, opts.rv1)
     
      for i = 1,n do
         local mse2 = nn.MSECriterion()
         parallelCriterion:add(mse2, opts.rv2 / n)
      end
     local mse3 = nn.MSECriterion()
     parallelCriterion:add(mse3,opts.rv3)
     return parallelCriterion 

   end
end

function createTrainableNetwork(opts)
   local model = nn.Sequential()

   model:add(nn.SpatialReflectionPadding(40, 40, 40, 40))
   model:add(nn.SpatialConvolution(3, 32, 9,9))
   model:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2))
   model:add(nn.SpatialConvolution(64, 128, 3,3,2,2))


   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())

   model:add(nn.SpatialConvolution(128, 128, 3,3))
   model:add(nn.SpatialBatchNormalization(128))
   model:add(nn.ReLU())


   model:add(nn.SpatialFullConvolution(128, 64, 3,3 ,2,2))
   model:add(nn.SpatialFullConvolution(64, 32, 3,3 ,2,2))
   model:add(nn.SpatialFullConvolution(32, 3, 10, 10))
   model:add(nn.Sigmoid())
   model:add(nn.AddConstant(-0.5, true))
   model:add(nn.MulConstant(255, true))
   return model
 end
 
 local function preprocessImage(input_image)
  
   input_image = input_image * 255
 
   input_image[{{1},{1},{},{}}] = input_image[{{1},{1},{},{}}] - 103.939
   input_image[{{1},{2},{},{}}] = input_image[{{1},{2},{},{}}] - 116.779
   input_image[{{1},{3},{},{}}] = input_image[{{1},{3},{},{}}] - 123.68

   aux = input_image:clone()
   aux[{{1},{1},{},{}}] = input_image[{{1},{3},{},{}}]
   aux[{{1},{3},{},{}}] = input_image[{{1},{1},{},{}}]
   return aux
 end 
 
 local function switchChannels(input_image)
   aux = input_image:clone()
   aux[{{1},{1},{},{}}] = input_image[{{1},{3},{},{}}]
   aux[{{1},{3},{},{}}] = input_image[{{1},{1},{},{}}]

   return aux
      
 end

 local function zeros_table(n, table_size)
   zeros = {}
   for i = 1 , n do
      local z = torch.zeros(table_size[i]:size())
      if params.gpu ~= -1 then
          z = z:cuda()
      end
      table.insert(zeros, z)
   end
   return zeros
 end

function getGramLayers(opts)
   local outputs = {}
   local gram_string = tostring(opts.gram_layers)
   for i = 1, #gram_string do
      outputs[#outputs + 1] = tonumber(gram_string:sub(i,i))
   end
   return outputs
end
 
 local function normalise(x)

   minim = x:min()
   maxim = x:max()
   x = 255 * (x - minim) / (maxim - minim) - 128
   return x
 end
 -- globals ---------------------------------------------------------------------

function getOptimizer(opts)
   optimizer = nil 
   if opts.optimizer == "adam" then
      optimizer = optim.adam
   end
   return optimizer
end

-- globals ---------------------------------------------------------------------
outputs = getGramLayers(params) 
no_net_outputs =  #outputs


conv_ix  = tonumber(params.conv_ix)
model = createTrainableNetwork()

if gpu ~= -1 then
   model = model:cuda()
end

-- references to model params and gradients
parameters, gradParameters = model:getParameters()

criterion = getCriterion(params,no_net_outputs) 

-- create input image and resize it to fit as network input

input_image_style    = image.load(params.image_style, 3,'double')
input_image_content  = image.load(params.image_content, 3, 'double')


sz = tonumber(params.img_size)
input_image_style    = image.scale(input_image_style, sz, sz):resize(1,3,sz, sz)
input_image_content  = image.scale(input_image_content, sz, sz):resize(1, 3, sz, sz)

input_image_content  = preprocessImage(input_image_content)
input_image_style    = preprocessImage(input_image_style)
-- gram transformation
gram = nn.Gram()

-- pretrained network
vgg = getModifiedVGG(params) --getPretrainedModel(params)

print('model'..tostring(vgg))
-- optimzer strategy
optimizer = getOptimizer(params)

if params.gpu ~= -1 then
   criterion           = criterion:cuda()
   input_image_style   = input_image_style:cuda()
   input_image_content = input_image_content:cuda()
   gram:cuda()
   vgg:cuda()
end

-- target response 
-- assume filter response is in second position of output dictionary

target_content          = vgg:forward(input_image_content)[conv_ix]:clone()
local vgg_input_style   = vgg:forward(input_image_style)
--target_style            = gram:forward(vgg_input_style)

local style_gram_input = {}
for i = 1 , no_net_outputs do
   table.insert(style_gram_input, vgg_input_style[outputs[i]]:clone())
end
   
-- style with gram
local gramTable = nn.ParallelTable()
for i = 1 , no_net_outputs do
   gramTable = gramTable:add(nn.Gram())
end

target_style      = gramTable:forward(style_gram_input) 
target_combined   = {target_content}
target_lbfgs      = {target_content}
for i = 1 , no_net_outputs do 
   table.insert(target_combined, target_style[i])
   table.insert(target_lbfgs, target_style[i])
end

table.insert(target_lbfgs,input_image_content)
--------------------------------------------------------------------------------
-- used windows
w1 = nil 
w2 = nil
w3 = nil
w4 = nil

if params.show_images == true then
   w1 = qtwidget.newwindow(224,224, params.label .. "|ic")
   w2 = qtwidget.newwindow(224,224, params.label .. "|is")
   w3 = qtwidget.newwindow(224,224, params.label .. "|o")
end

-- function that generates gradients needed by optim to update parameters
-- style part
local feval_combined = function(newParameters)

   if parameters ~= newParameters then parameters:copy(newParameters) end

   gradParameters:zero()

   if gram.gradInput then
      gram.gradInput:zero()
      gram.output:zero()
   end

   vgg:zeroGradParameters()
   -- trecem prin reteaua de transformare
   local outputModel  = model:forward(input_image_content)
   -- vgg returneaza un tabel cu 5 rezultate de la diferite nivele
   local all_vgg_outs = vgg:forward(outputModel)

   --local out_vgg_ = vgg:forward(outputModel)
   local out_vgg_ = all_vgg_outs
   
   
   -- stilul foloseste doar nivelele cu indice in outputs
   local style_gram_input = {}
   for i = 1 , no_net_outputs do
      table.insert(style_gram_input, all_vgg_outs[outputs[i]])
   end

   -- content foloseste doar un nivel, cel din conv_ix
   local out_vgg_content = out_vgg_[conv_ix]:clone()

   -- style with gram
   local gramTable = nn.ParallelTable()
   for i = 1 , no_net_outputs do
      gramTable = gramTable:add(nn.Gram())
   end

   gram_vgg = gramTable:forward(style_gram_input) 
   local combined_output = {out_vgg_content}
   for i = 1, no_net_outputs do
      table.insert(combined_output, gram_vgg[i])   
   end

   local loss = criterion:forward(combined_output , target_combined)
   if params.show_loss then
      print(loss)
   end
   
   -- backward phase
   
   local dLoss = criterion:backward(combined_output, target_combined)
   -- selectam doar lossurile de stil
   dLossStyle = {}
   for i = 1 , no_net_outputs do
      table.insert(dLossStyle, dLoss[i+1])
   end
   
   
   local dGramTable = gramTable:backward(style_gram_input, dLossStyle)

   -- gradientii pentru iesirile intermediare vgg care nu sunt folosite sunt zero
   gradsVgg = zeros_table(5, all_vgg_outs) 
   for i = 1 , no_net_outputs do
      gradsVgg[outputs[i]] = dGramTable[i]
   end
   -- adaugam gradientul de context
   gradsVgg[conv_ix] = gradsVgg[conv_ix] + dLoss[1]
 
   local gradOutput = vgg:backward(outputModel, gradsVgg)
 
   
   local ginput = model:backward(input_image_content, gradOutput)
   

   if params.show_images then
      w1 = image.display{image=switchChannels(input_image_content),
         offscreen=false, win=w1}
      w3 = image.display{image=switchChannels(outputModel),
         offscreen=false, win=w3}
  end

   if params.save_images == true and iterations % params.save_every == 0 then
      image.save(params.save_folder ..'/'.. params.label .. "_" ..
         tostring(iterations) .. ".jpg", switchChannels((outputModel + 128) / 255):squeeze())
      image.save(params.save_folder ..'/'.. params.label .. "_" ..
         tostring(iterations) .. "andu.jpg", switchChannels((outputModel)):squeeze())
   end


   return loss, gradParameters
end

-- function that generates gradients needed by lbfgs
local feval_combined_lbfgs = function(newParameters)

   if parameters_lbfgs ~= newParameters then parameters_lbfgs:copy(newParameters) end

   gradParameters_lbfgs:zero()

   if gram.gradInput then
      gram.gradInput:zero()
      gram.output:zero()
   end

   vgg:zeroGradParameters()

   local out_vgg_ = vgg:forward(parameters_lbfgs)--outputModel)
   local out_vgg_content = out_vgg_[conv_ix]:clone()
   
   local z1 = torch.zeros(out_vgg_[1]:size())
   local z2 = torch.zeros(out_vgg_[2]:size())
   local z3 = torch.zeros(out_vgg_[3]:size())
   local z4 = torch.zeros(out_vgg_[4]:size())
   local z5 = torch.zeros(out_vgg_[5]:size())
   if params.gpu ~= -1 then
       z1 = z1:cuda()
       z2 = z2:cuda()
       z3 = z3:cuda()
       z4 = z4:cuda()
       z5 = z5:cuda()
   end

  
     -- style with gram
   local out_vgg_style = gram:forward(out_vgg_[conv_ix])


   local style_gram_input = {}
   for i = 1 , no_net_outputs do
      table.insert(style_gram_input, out_vgg_[outputs[i]])
   end
 
   -- style with gram
   local gramTable = nn.ParallelTable()
   for i = 1 , no_net_outputs do
      gramTable = gramTable:add(nn.Gram())
   end

   gram_vgg = gramTable:forward(style_gram_input) 

   local combined_output = {out_vgg_content}
   for i = 1, no_net_outputs do
      table.insert(combined_output, gram_vgg[i])   
   end

   table.insert(combined_output, parameters_lbfgs)

   local loss = criterion:forward(combined_output , target_lbfgs)

   if params.show_loss then
      print(loss)
   end
   
   -- backward phase
   
   local dLoss = criterion:backward(combined_output, target_lbfgs)
   
   -- selectam doar lossurile de stil
   dLossStyle = {}
   for i = 1 , no_net_outputs do
      table.insert(dLossStyle, dLoss[i+1])
   end
   
   
   local dGramTable = gramTable:backward(style_gram_input, dLossStyle)
   -- gradientii pentru iesirile intermediare vgg care nu sunt folosite sunt zero
   gradsVgg = zeros_table(5, out_vgg_) 
   for i = 1 , no_net_outputs do
      gradsVgg[outputs[i]] = dGramTable[i]
   end
   -- adaugam gradientul de context
   gradsVgg[conv_ix] = gradsVgg[conv_ix] + dLoss[1]
 
   local gradOutput = vgg:backward(outputModel, gradsVgg)
 
   gradOutput = gradOutput + dLoss[no_net_outputs+2]

   
   parameters_lbfgs = normalise(parameters_lbfgs) 
   if params.show_images then
      w1 = image.display{image=switchChannels(input_image_content), offscreen=false, win=w1}
      w2 = image.display{image=switchChannels(input_image_style),
         windowTitle=params.label .. "input_image_style", offscreen=false, win=w2}
      w3 = image.display{image=switchChannels(parameters_lbfgs), offscreen=false, win=w3}
   end


   if params.save_images then
      filename = params.save_folder.. '/' .. params.label .. "output" .. tostring(iterations) .. ".jpg"
      image.save(filename,
         switchChannels((parameters_lbfgs + 128)/ 255):squeeze())
   end


   return loss, gradOutput
end

batch_index = 0
batchsize = 4 
function loadData()
   --batchsize = 10 
   folder  = '/data/VOC/VOCdevkit/VOC2012/JPEGImages/' 
   batch = torch.zeros(batchsize,3,params.img_size,params.img_size);

   local pl = require('pl.import_into')()
   local t = {}
   local data_size = 0
   for i,f in ipairs(pl.dir.getallfiles(folder, '*.jpg')) do
      t[i] = f
      data_size = data_size + 1
      --t[i] = { f, pl.path.basename(pl.path.dirname(f)) }
   end
  for i=1,batchsize do
      local index = batch_index * batchsize + i
      img = image.load(t[index], 3, 'double')
      img = image.scale(img,params.img_size,params.img_size):resize(1,3,
         params.img_size, params.img_size)
      batch[i] = preprocessImage(img)
      --if i == batchsize then
      --   break
      --end
   end
   batch_index = batch_index + 1
   if (batch_index + 1) * batchsize > data_size then
      batch_index = 0
   end
   print('batch index = '..batch_index)
   return batch 

end

function save_input()
  if params.save_images == true then
      local save_str = params.save_folder .. "/input_image.jpg"
      print("saving input image to " .. save_str .. "...")

      image.save(params.save_folder .. "/input_image.jpg",
         switchChannels((input_image_style+128)/255):squeeze())

      image.save(params.save_folder .. "/input_image.jpg",
               switchChannels((input_image_content+128) / 255):squeeze())
   end
end

iterations = 0
function train()
   save_input()
   while true do
      iterations = iterations + 1 
      print("Starting iteration " .. tostring(iterations) ..  "..")
      -- choose evaluation function for optimizer
      func = feval_content
      if params.train_type == 'net-style' then
         func = feval_style
      elseif params.train_type == 'netcombined' then
         func = feval_combined
      end
      optimizer(func, parameters, optimParams)
      if iterations % (100*params.save_every) == 0 then
         print("saving model...")
         torch.save(params.save_folder..'/'..tostring(iterations)..'_'..params.label..'_'..params.model_name, model)
      end
   end
end
iterations = 0
parameters_lbfgs     = torch.rand(input_image_content:size()) * 255
gradParameters_lbfgs = torch.zeros(parameters_lbfgs:size())

if params.gpu ~= -1 then 
   parameters_lbfgs     = parameters_lbfgs:cuda()
   gradParameters_lbfgs = gradParameters_lbfgs:cuda()
end

mse = nn.MSECriterion()

function train_lbfgs()
   save_input()
   while true do
      iterations = iterations + 1
      print("Running Iteration: " .. tostring(iterations) .. "...")
   
      optimizer = optim.lbfgs
      optimizer(feval_combined_lbfgs, parameters_lbfgs, optimParams)
      if iterations % (100*params.save_every) == 0 then
         print("saving model...")
         torch.save(params.save_folder..'/'..tostring(iterations)..'_'..params.label..'_'..params.model_name, model)
      end
 
   end
end

function trainBatch()
   save_input()
   i = 1
   batch = loadData()
   print(batch:size())
   if params.gpu ~= -1 then
      batch = batch:cuda()
   end
   iterations = 0
   while true do
      iterations = iterations + 1
      print("Iteration:" .. tostring(iterations))
      i = iterations % batchsize + 1
      input_image_content   = batch[{{i},{},{},{}}]
      --[[
      if iterations % 10 == 0 then
         print('save '.. tostring(iterations))
         image.save("./content_images/img_"..tostring(iterations)..'.jpg',
                           switchChannels((input_image_content + 128)/255):squeeze())
      end
      --]]

      target_content        = vgg:forward(input_image_content)[conv_ix]:clone()
      target_combined[1] = target_content
      
      func = feval_content
      if params.train_type == 'net-style' then
         func = feval_style
      elseif params.train_type == 'netcombined' then
         func = feval_combined
      end
      optimizer(func, parameters, optimParams)
      
      if iterations % batchsize == (batchsize - 1) then
            print('Load another batch')
            batch = loadData()
            if params.gpu ~= -1 then
               batch = batch:cuda()
            end
      end
      
      if iterations % (5*params.save_every) == 0 then
         print("saving model...")
         torch.save(params.save_folder..'/'..tostring(iterations)..'_'..params.label..'_'..params.model_name, model)
      end
      
   end
end
gpu = tostring(params.gpu)
print(params.train_type)
if params.train_type == "lbfgs" then
   train_lbfgs()
elseif params.train_type == "netcombined" then
   print('...... Start training Net')
   trainBatch()
--     train()
else
   print("Unknown train procedure")
end
