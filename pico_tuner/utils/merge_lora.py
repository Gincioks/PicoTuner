# import logging
# import sys
# # import shutil

# from .llama import add_lora

# import os


# logging.basicConfig(format='%(asctime)s %(message)s',
#                     level=logging.INFO, filename='logs/merge_lora.log')

# model_path = sys.argv[1]
# lora_path = sys.argv[2]
# out_model_path = sys.argv[3]

# # shutil.copytree(model_path, out_model_path)
# add_lora(out_model_path, lora_path)
