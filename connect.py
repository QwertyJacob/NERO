import os
import wandb
import sys
import types
import importlib

# Connect to internet
os.environ['HTTP_PROXY'] = 'http://proxy.uninsubria.it:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy.uninsubria.it:3128/'

# Login to wandb
wandb.login()

def reload_modules(list_of_modules):
    for module in list_of_modules:
        print(f'reloading module :{module}')
        importlib.reload(module)