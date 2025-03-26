from EC57.ec57_script import del_result, ec57_eval
import os
import sys

physionet_directory = "/home/itr/Project/rebar/data/ecg_segmentation/"
sys.path.append(physionet_directory)

output_ec57_directory = os.path.dirname(os.path.abspath(__file__)) + "/output/"

ec57_eval('mit-bih-arrhythmia-database-1.0.0',          
          output_ec57_directory=output_ec57_directory,          
          physionet_directory=physionet_directory,          
          beat_ext_db='atr',          
          event_ext_db=None,          
          beat_ext_ai='beat',          
          event_ext_ai=None)


del_result(dir_db = 'mit-bih-arrhythmia-database-1.0.0',         
             physionet_directory=physionet_directory,           
             output_ec57_directory=output_ec57_directory)