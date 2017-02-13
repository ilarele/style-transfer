--------------------------------------------------------------------------------
-- Load command line options
-- Lateral effect is the population of the "params" global dictionary 
--------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Test network for style transfer')
cmd:text()
cmd:text('Options')
cmd:option('--model_path', "/data/saved_transfer_models/20000_exp_netcombined_full_model.t7", 'path to model to be loaded')
cmd:option('--image_path', "input_images/content_chair.jpg", "image on which we should transfer style")
cmd:option('--save_path',  "transferred_images/out.jpg", "path where style transfered images are saved")
cmd:option('--show_images', false, "display saved image via qt container")
cmd:option('--gpu',-1,'gpu to be used')


-- parse input params
params= cmd:parse(arg)
