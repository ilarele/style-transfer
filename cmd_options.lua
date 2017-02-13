--------------------------------------------------------------------------------
-- Load command line options
-- Lateral effect is the population of the "params" global dictionary 
--------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train network for style transfer')
cmd:text()
cmd:text('Options')
cmd:option('--train_type', "netcombined", 'desired train procedure')
cmd:option('--vgg_truncation',17,'num of layers of truncation should be perf.')
cmd:option('--vgg_str',"VGG_ILSVRC_19_layers", 'vgg model to be used')
cmd:option('--gpu',-1,'gpu to be used')
cmd:option('--optimizer',"adam",'optimizer strategy to be used')
cmd:option('--show_images',false, "show images via a qt container")
cmd:option('--save_images',true, "save images to files")
cmd:option('--show_loss', true, "show the loss at each iteration")

-- criterion hyperparameters
cmd:option('--rv1', 1, "regulate hyperparameter 1")
cmd:option('--rv2', 0.000001, "regulate hyperparameter 2")
cmd:option('--rv3', 0.0, "regulate hyperparameter 3, used in lbfgs")


cmd:option('--label', 'default_label', "label for this run")

cmd:option('--save_folder', "default_save_folder/", "deflt folder to save imgs")

-- used image names
cmd:option('--image_content', "input_images/sky.jpg", "content image")
cmd:option('--image_style', "input_images/starry_night.jpg", "style image")

cmd:option('--save_every', 1, 'save at every n iterations')
cmd:option('--model_name', 'model.t7', 'save model with this filename')
cmd:option('--img_size',224, "image height")
cmd:option('--conv_ix',1, "convolution index")
cmd:option('--gram_layers',123, "gram layers used for style")




-- parse input params
params = cmd:parse(arg)
