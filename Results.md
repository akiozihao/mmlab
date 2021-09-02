
| Model                 | Private Half | Public Half | Private Test |
|-----------------------|--------------|-------------|--------------|
| mot17_fulltrain       |   69.8, 70.2 |  68.1, 66.8 |         67.3 |
| mot17_half            |   64.2, 66.1 |  63.2, 63.1 |        -     |
| crowdhuman            |   53.8, 52.2*|  52.7, 50.7*|        -     |
| mmlab_full            |   68.5, 68.2 |  66.2, 65.7 |        -     |
| mmlab_half            |   ----, ---- |  ----, ---- |        -     |

(*) python test.py tracking --exp_id crowdhuman --pre_hm --test_dataset mot --dataset_version 17halfval --track_thresh 0.6 --resume --ltrb_amodal --input_h 544 --input_w 960