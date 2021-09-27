| Exp        | CheckPoint | Trained With | Tested With | DCN | Gaussian | Age | Private Half | Public Half | Private Test |
|------------|------------|--------------|-------------|-----|----------|-----|--------------|-------------|--------------|
| Fulltrain  | original   | O            | O           | O   | O        |     | 69.8, 70.2   | 68.1, 66.8  | 67.3         |
| Fulltrain  | original   | O            | O           | O   | O        | 3   | 67.0, 70.9   | ----, ----  | -            |
| Halftrain  | original   | O            | O           | O   | O        |     | 64.2, 66.1   | 63.2, 63.1  | -            |
| Halftrain  | original   | O            | O           | O   | O        | 3   | 65.2, 66.1   | ----, ----  | -            |
| crowdhuman | dla        | O            | O           | O   | O        |     | 53.8, 52.2*  | 52.7, 50.7* | -            |
| Fulltrain  | original   | O            | M           | O   | O        |     | 70.0, 70.2   | ----, ----  | -            |
| Fulltrain  | original   | O            | M           | O   | O        | 3   | 71.6, 70.5   | ----, ----  | -            |
| Fulltrain  | original   | O            | M           | O   | O        | 5   | 71.6, 70.5   | ----, ----  | -            |
| Fulltrain  | original   | O            | M           | M   | O        |     | 70.0, 70.2   | ----, ----  | -            |
| Fulltrain  | original   | O            | M           | M   | O        | 3   | 71.6, 70.5   | ----, ----  | -            |
| Fulltrain  | original   | O            | M           | M   | O        | 5   | 71.6, 70.5   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | O   | O        |     | 64.4, 66.8   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | O   | O        | 3   | 65.3, 67.0   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | O   | O        | 5   | 65.5, 67.1   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | M   | O        |     | 64.4, 66.8   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | M   | O        | 3   | 65.3, 67.0   | ----, ----  | -            |
| Halftrain  | original   | O            | M           | M   | O        | 5   | 65.5, 67.1   | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | O   | O        |     | 57.1, 50.9   | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | O   | O        |     | 54.7, 54.3** | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | M   | O        |     | 57.1, 50.9   | ----, ----  | -            |
| CrowdHuman | dla        | O            | M           | M   | O        |     | 54.7, 54.3** | ----, ----  | -            |
| CrowdHuman | dla        | M            | M           | O   | O        |     | 47.4, 57.4   | ----, ----  | -            |
| CrowdHuman | dla        | M            | M           | O   | O        |     | 43.0, 50.1** | ----, ----  | -            |
| Halftrain  | original   | M            | M           | O   | O        | 3   | 65.7, 66.2   | ----, ----  | -            |
| Halftrain  | trained MO | M            | M           | M   | O        |     | 63.2, 65.3   | ----, ----  | -            |


(*) python test.py tracking --exp_id crowdhuman --pre_hm --test_dataset mot --dataset_version 17halfval --track_thresh 0.6 --resume --ltrb_amodal --input_h 544 --input_w 960

(**) model.tracker.obj_score_thr=0.6
