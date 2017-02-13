require 'loadcaffe'
require 'image'
require 'nn'
require "optim"

require "cmd_options_test"

if params.show_images == true then
   require 'qtwidget'
end

require "setup"


 
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

model = torch.load(params.model_path)

if params.gpu ~= -1 then
   model = model:cuda()
else
   model = model:float()
end

sz = 500 

im = image.load(params.image_path, 3, "double")

im = image.scale(im, sz, sz):resize(1,3, sz, sz)

im = preprocessImage(im)

if params.gpu ~= -1 then
   im = im:cuda()
else
   im = im:float()
end

im2 = model:forward(im)

im2 = switchChannels(im2)

image.save(params.save_path, im2:squeeze())

if params.show_images == true then
   image.display(im2:squeeze())
end

