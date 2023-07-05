#!/bin/sh
cd ~/Desktop/ProRoc/DeepSDF
python3 test_json.py --to_obj
echo "test_json.py --to_obj done."
python3 preprocess_data.py --data_dir data_PPRAI --source Dataset_PPRAI/ --name Dataset_PPRAI --split examples/splits/PPRAI_test_golem.json --test --skip --threads 15
echo "preprocess done."
python3 test_json.py --npz_th
echo "test_json.py --npz_th done."
python3 reconstruct.py -e examples/PPRAI -c latest --split examples/splits/PPRAI_test_golem.json -d data_PPRAI
echo "reconstruct done."
python3 test_json.py --to_pcd
echo "test_json.py --to_obj done."