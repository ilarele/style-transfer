################################################################################
## Script that runs a set of experiments for style transfer
################################################################################
import sys
import subprocess
import os
import os.path
import operator
    
run_command = "th train.lua "

lbfgs_order = [\
        "--train_type ", " --label ", " --rv1 ", " --rv2 ", " --rv3 ", \
        " --image_content ", " --image_style ", " --save_folder ", \
        " --save_every ",\
        " --gpu ", " --conv_ix ", " --gram_layers"]

processes = set()
max_processes = 30



# label rv1 rv2 rv3 image1 image2 save_folder
def treatLine(csv_args):
   global run_command
   call_list = [x[0] + " " +  x[1] + " " for x in zip(lbfgs_order, csv_args)]

   call_string = reduce(operator.add, call_list,"")

   if "--show_images" in sys.argv:
      call_string = call_string + " --show_images "
      run_command = "qlua train.lua "

   if "--noterm" in sys.argv:
       run_command = "nohup " + run_command
       call_string = call_string + " & disown"
 

   call_string = run_command + call_string

   print("Command to be executed " + "[" + call_string + "]")

   save_folder = csv_args[7]
   if not os.path.exists(save_folder):
      print("Could not find path, to save folder " + save_folder + \
            " , creating it now...")
      os.makedirs(save_folder) 

   print("running command...")
   if not ("--noterm" in sys.argv):
       os.system(call_string)
   else:
      p = subprocess.Popen(call_string, shell=True, \
               stdout=open("outerrs/" + csv_args[1] + ".out", 'w'),\
               stderr=open("outerrs/" + csv_args[1] + ".err", 'w'))
      print("process " + str(p) + " started.")

      processes.add(p)


#this is to be filled
run_args = []

if len(sys.argv) < 2:
    print("Usage: python run_scripts.py <config_file>")
    sys.exit(0)

config_file = sys.argv[1]
if not os.path.isfile(config_file):
    print("Usage: python run_scripts.py <config_file>")
    print("Config file given is not valid!")
    sys.exit(0)

with open(config_file) as f:
    file_content = f.read()
    experiments = file_content.split("\n")
    for exp in experiments:
        if exp == "":
            break
        if len(processes) > max_processes:
            print("reached max num of processes")
            break
        csv_args = exp.split(",")
        argname = csv_args[0]
        if argname == "lbfgs" or argname == "netcombined":
            treatLine(csv_args)
        else:
            print("unknown command")
