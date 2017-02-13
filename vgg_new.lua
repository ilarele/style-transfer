require 'loadcaffe'
require 'image'
require 'nn'
require "optim"
require "gram"
require "nngraph"



function createVGG(opts, vgg)
   local X = nn.Identity()()
   local conv1_1 = nn.SpatialConvolution(3,64,3,3,1,1,1,1)(X)
   local relu1_1 = nn.ReLU()(conv1_1)
   ----[[
   local conv1_2 = nn.SpatialConvolution(64,64,3,3,1,1,1,1)(relu1_1)
   local relu1_2 = nn.ReLU()(conv1_2)
   local pool1 = nn.SpatialMaxPooling(2,2,2,2)(relu1_2)

   local conv2_1 = nn.SpatialConvolution(64,128,3,3,1,1,1,1)(pool1)
   local relu2_1 = nn.ReLU()(conv2_1)
   local conv2_2 = nn.SpatialConvolution(128,128,3,3,1,1,1,1)(relu2_1)
   local relu2_2 = nn.ReLU()(conv2_2)
   local pool2 = nn.SpatialMaxPooling(2,2,2,2)(relu2_2)
 
   local conv3_1 = nn.SpatialConvolution(128,256,3,3,1,1,1,1)(pool2) 
   local relu3_1 = nn.ReLU()(conv3_1)
   local conv3_2 = nn.SpatialConvolution(256,256,3,3,1,1,1,1)(relu3_1)
   local relu3_2 = nn.ReLU()(conv3_2)
   local conv3_3 = nn.SpatialConvolution(256,256,3,3,1,1,1,1)(relu3_2)
   local relu3_3 = nn.ReLU()(conv3_3)
   local conv3_4 = nn.SpatialConvolution(256,256,3,3,1,1,1,1)(relu3_3)
   local relu3_4 = nn.ReLU()(conv3_4)
   local pool3 = nn.SpatialMaxPooling(2,2,2,2)(relu3_4)

   local conv4_1 = nn.SpatialConvolution(256,512,3,3,1,1,1,1)(pool3)
   local relu4_1 = nn.ReLU()(conv4_1)
   local conv4_2 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu4_1)
   local relu4_2 = nn.ReLU()(conv4_2)
   local conv4_3 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu4_2)
   local relu4_3 = nn.ReLU()(conv4_3)
   local conv4_4 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu4_3)
   local relu4_4 = nn.ReLU()(conv4_4)
   local pool4 = nn.SpatialMaxPooling(2,2,2,2)(relu4_4)

 
   local conv5_1 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(pool4)
--   local relu5_1 = nn.ReLU()(conv5_1)
   --]]
--   local conv5_2 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu5_1)
--   local relu5_2 = nn.ReLU()(conv5_2)
--   local conv5_3 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu5_2)
--   local relu5_3 = nn.ReLU()(conv5_3)
--   local conv5_4 = nn.SpatialConvolution(512,512,3,3,1,1,1,1)(relu5_3)
--   local relu5_4 = nn.ReLU()(conv5_4)
--   local pool5 = nn.SpatialMaxPooling(2,2,2,2)(relu5_4)




   return nn.gModule({X}, {conv1_1, conv2_1, conv3_1, conv4_1, conv5_1}) 
end


function createVGG0(opts, vgg)
   local X = nn.Identity()()
   local conv1 = nn.SpatialConvolution(3,96,7,7,1,1)(X)
   local relu1 = nn.ReLU()(conv1)
   local scmlrn = nn.SpatialCrossMapLRN(5, 0.0005, 0.75, 2)(relu1)
   local smp1 = nn.SpatialMaxPooling(3,3,3,3)(scmlrn)

   local conv2 = nn.SpatialConvolution(96, 256, 5,5)(smp1)

   local relu2 = nn.ReLU()(conv2)
   local smp2 = nn.SpatialMaxPooling(2,2, 2,2)(relu2)

   local conv3 = nn.SpatialConvolution(256, 512, 3,3, 1,1, 1,1)(smp2)
   local relu3 = nn.ReLU()(conv3)

   local conv4 = nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1)(relu3)
   local relu4 = nn.ReLU()(conv4)

   local conv5 = nn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1)(relu4)
   local relu5 = nn.ReLU()(conv5)

   --local smp3 = nn.SpatialMaxPooling(3,3, 3,3)(relu5)

   return nn.gModule({X}, {relu5}) 
end


function truncateModel(model, opts)
   print(model)
   for i = 1,tonumber(params.vgg_truncation) do
      model:remove(#model.modules)
   end
   print(model)

   return model 
end


function getPretrainedModel(opts)
   
   local gpu_str = "nn"
   if gpu ~= -1 then
      gpu_str = "nn" -- TODO investigate actual implication
   end

   local model = loadcaffe.load('vgg/deploy_vgg19.prototxt',
   'vgg/'.. opts.vgg_str .. '.caffemodel', gpu_str)

   model = truncateModel(model, opts)

   return model
end

function getModifiedVGG(opts)
   local m = getPretrainedModel(opts)
   local new_vgg = createVGG(opts)

   local pm, gd = m:getParameters()
   local vp, vd = new_vgg:getParameters()

   vp[{}] = pm[{}]

   return new_vgg

end

