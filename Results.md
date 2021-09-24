| Exp        | CheckPoint | Trained With | Tested With | DCN | Gaussian | Age | Private Half | Public Half | Private Test |
|------------|------------|--------------|-------------|-----|----------|-----|--------------|-------------|--------------|
| Fulltrain  | original   | O            | O           | O   | O        |     | 69.8, 70.2   | 68.1, 66.8  | 67.3         |
| Fulltrain  | original   | O            | O           | O   | O        | 3   | 67.0, 70.9   | ----, ----  | -            |
| Halftrain  | original   | O            | O           | O   | O        |     | 64.2, 66.1   | 63.2, 63.1  | -            |
| Halftrain  | original   | O            | O           | O   | O        | 3   | 65.2, 66.1   | 63.2, 63.1  | -            |
| crowdhuman | dla        | O            | O           | O   | O        |     | 53.8, 52.2*  | 52.7, 50.7* | -            |
| Fulltrain  | original   | O            | M           | M   | O        |     | 68.5, 68.2   | 66.2, 65.7  | -            |
| Fulltrain  | original   | O            | M           | M   | O        | 3   | 67.0, 71.2   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | M   | O        |     | ----, ----   | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | M   | O        |     | 58.2, 51.3   | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | M   | O        |     | 56.5, 54.9** | ----, ----  | -            |
| Halftrain  | original   | M            | M           | O   | O        | 3   | 65.7, 66.2   | ----, ----  | -            |
| Halftrain  | trained MO | M            | M           | M   | O        |     | 63.2, 65.3   | ----, ----  | -            |
| Halftrain  | mot17_half | M            | M           | O   | O        | 3   | 65.5, 67.1   | ----, ----  | -            |


(*) python test.py tracking --exp_id crowdhuman --pre_hm --test_dataset mot --dataset_version 17halfval --track_thresh 0.6 --resume --ltrb_amodal --input_h 544 --input_w 960

(**) model.tracker.obj_score_thr=0.6
