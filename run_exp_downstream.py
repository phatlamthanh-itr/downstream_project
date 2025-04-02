import argparse
import torch
import os
import numpy as np
import subprocess
from utils.utils import printlog, load_data, import_model, init_dl_program
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
from downstream.downstream_config import downstream_config
from eval_downstream import eval_beat_segmentation, post_processing
from downstream.downstream_nets import LSTMDecoder2500, LSTMDecoder, ResNetLSTM
all_expconfigs = {**allrebar_expconfigs}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str, required=True)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    args = parser.parse_args()

    # selecting config according to arg
    config = all_expconfigs[args.config]
    config.set_rundir(args.config)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", config.data_name, config.run_dir, "checkpoint_best.pkl"))):
        train_data, _, val_data, _, _, _ = load_data(config = config, data_type = "fullts")
        model = import_model(config, train_data=train_data, val_data=val_data)
        model.fit()

# ==================== RUN DOWNSTREAM TASKS ==============================

    # Load dataset for downstream tasks segmentation
    train_data, train_labels, val_data, val_labels, _, _  = load_data(config = config, data_type = "subseq", label_size=downstream_config.label_size, downstream=True)
    model = import_model(config, reload_ckpt = True)

    run_dir = model.run_dir

    # Handle feature extraction from SSL model
    # if os.path.exists(os.path.join(run_dir, "encoded_train.pth")):
    #     train_features = torch.load(os.path.join(run_dir, "encoded_train.pth"))
    #     val_features = torch.load(os.path.join(run_dir, "encoded_val.pth"))
    # else:
    train_features = model.encode(train_data)
    val_features = model.encode(val_data)
    printlog("Encoding train, val, test ...", run_dir)
    # torch.save(train_features, os.path.join(run_dir, "encoded_train.pth"), pickle_protocol=4)
    # torch.save(val_features, os.path.join(run_dir, "encoded_val.pth"), pickle_protocol=4)
    if downstream_config.label_size == 2500:
        print("Load 2500-time step data and model ....")
        # downstream_model = LSTMDecoder2500(input_seq_length=downstream_config.subseq_size, feature_dim=downstream_config.feature_dim
        #                                    , hidden_dim = downstream_config.hidden_dim, num_layers = downstream_config.num_layers,
        #                                    output_length=downstream_config.output_size, bidirectional=True)
        downstream_model = ResNetLSTM(input_seq_length=downstream_config.subseq_size, feature_dim=downstream_config.feature_dim, 
                                hidden_dim=downstream_config.hidden_dim, 
                                num_layers = downstream_config.num_layers, output_length=downstream_config.output_size)

    else:
        print("Load 80-time step data and model ....")
        downstream_model = LSTMDecoder(input_seq_length=downstream_config.subseq_size, feature_dim=downstream_config.feature_dim,
                                    hidden_dim=downstream_config.hidden_dim, num_layers=downstream_config.num_layers,
                                    output_length=downstream_config.output_size, bidirectional=True)


    best_val_loss, best_val_acc = eval_beat_segmentation(downstream_model=downstream_model, train_data=train_features, train_labels = train_labels, 
                           val_data=val_features, val_labels = val_labels)

    print("-------Downstream training done --------------")
    print("Best val loss: ", best_val_loss, "Best val acc: ", best_val_acc)
    

    print("Load model prediction! Start post-processing .....")


    post_processing(label_size=downstream_config.label_size,pred_path=downstream_config.save_pred_dir, save_beat_dir=downstream_config.save_beat_dir, 
                    ecgpath="data/ecg_segmentation", visualize=False)


 

              





