### SPIRAL ###
#### Ensemble ####
python3 train.py --gpu 3 --dataset spiral --n_tails 20 --method ensemble

#### Dropout ####
python3 train.py --gpu 3 --dataset spiral --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt --n_tails_teacher 20 --n_tails 20 --method drop

#### Gaussian ####
python3 train.py --gpu 3 --dataset spiral --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt --n_tails_teacher 20 --n_tails 20 --method gauss

#### Hydra ####
python3 train.py --gpu 3 --dataset spiral --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt --n_tails_teacher 20 --n_tails 20 --method hydra

#### EnDD #### 
python3 train.py --gpu 3 --dataset spiral --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt --n_tails_teacher 20 --n_tails 20 --method endd

#### Hydra+ ####
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt --n_tails_teacher 20 --n_tails 20
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt  --n_tails_teacher 20 --n_tails 10
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt  --n_tails_teacher 20 --n_tails 5

python3 compare_outputs.py --folder_paths ./spiral/spiral-teacher-20-20220425-105510 ./spiral/spiral-gauss-20-20220509-122117/ ./spiral/spiral-drop-20-20220501-153122 ./spiral/spiral-endd-20-20220501-153324 ./spiral/spiral-hydra-20-20220508-165930 ./spiral/spiral-hydra+-20-20220508-155008 --labels Ensemble Gauss Drop EnDD Hydra Hydra+ --rows 2 --columns 3

python3 compare_outputs.py --folder_paths ./spiral/spiral-teacher-20-20220425-105510/ ./spiral/spiral-hydra+-20-20220508-155008 ./spiral/spiral-hydra+-10-20220508-160752 ./spiral/spiral-hydra+-5-20220508-161203 --labels Ensemble Hydra+-20 Hydra+-10 Hydra+-5 --dataset spiral --rows 1 --columns 4

#### Hydra+ ablation ####
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"beta_scheduler":1.0, "lambda_scheduler": 0.0}' --label no_distance_no_loss
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"beta_scheduler":1.0}' --label yes_distance_no_loss
python3 train.py --gpu 3 --dataset spiral --method hydra+  --load_teacher ./spiral/spiral-teacher-20-20220425-105510/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"lambda_scheduler": 0.0}' --label no_distance_yes_loss

python3 compare_outputs.py --folder_paths ./spiral/spiral-hydra+-20-20220508-155850-no_distance_no_loss ./spiral/spiral-hydra+-20-20220508-160019-yes_distance_no_loss ./spiral/spiral-hydra+-20-20220508-160145-no_distance_yes_loss ./spiral/spiral-hydra+-20-20220508-155008 --dataset spiral --labels "$\lambda=0.0,\beta=1.0$" "$\lambda=4.0,\beta=1.0$" "$\lambda=0.0,\beta=0.5$" "$\lambda=4.0,\beta=0.5$"  --rows 1 --columns 4

### REGRESSION ### 
#### Ensemble ####
python3 train.py --gpu 3 --dataset regress --n_tails 20 --method ensemble

#### Dropout ####
python3 train.py --gpu 3 --dataset regress --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt --n_tails_teacher 20 --n_tails 20 --method drop 

#### Gaussian ####
python3 train.py --gpu 3 --dataset regress --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt --n_tails_teacher 20 --n_tails 20 --method gauss 

#### Hydra ####
python3 train.py --gpu 3 --dataset regress --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt --n_tails_teacher 20 --n_tails 20 --method hydra 

#### Hydra+ ####
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt --n_tails_teacher 20 --n_tails 20
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt  --n_tails_teacher 20 --n_tails 10
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt  --n_tails_teacher 20 --n_tails 5

python3 compare_outputs.py --folder_paths ./regress/regress-teacher-20-20220501-201652/ ./regress/regress-gauss-20-20220509-122316/ ./regress/regress-drop-20-20220502-190516/  ./regress/regress-hydra-20-20220508-170405/ ./regress/regress-hydra+-20-20220508-110411 --labels Ensemble Gauss Drop Hydra Hydra+ --dataset regress --rows 2 --columns 3
python3 compare_outputs.py --folder_paths ./regress/regress-teacher-20-20220501-201652/ ./regress/regress-hydra+-20-20220508-110411/ ./regress/regress-hydra+-10-20220508-121326/ ./regress/regress-hydra+-5-20220508-120041/ --labels Ensemble Hydra+-20  Hydra+-10 Hydra+-5 --dataset regress --rows 1 --columns 4

#### Hydra+ ablation ####
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"beta_scheduler":1.0, "lambda_scheduler": 0.0}' --label no_distance_no_loss
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"beta_scheduler":1.0}' --label yes_distance_no_loss
python3 train.py --gpu 3 --dataset regress --method hydra+  --load_teacher ./regress/regress-teacher-20-20220501-201652/weights.pt  --n_tails_teacher 20 --n_tails 20 --hyperparameters '{"lambda_scheduler": 0.0}' --label no_distance_yes_loss

python3 compare_outputs.py --folder_paths ./regress/regress-hydra+-20-20220508-111538-no_distance_no_loss ./regress/regress-hydra+-20-20220508-111640-yes_distance_no_loss ./regress/regress-hydra+-20-20220508-111837-no_distance_yes_loss regress/regress-hydra+-20-20220508-110411 --dataset regress --labels "$\lambda=0.0,\beta=1.0$" "$\lambda=2e^{-3},\beta=1.0$" "$\lambda=0.0,\beta=0.5$" "$\lambda=2e^{-3},\beta=0.5$" --rows 1 --columns 4

### CIFAR-10 ###
#### Ensemble ####
python3 train.py --gpu 3 --dataset cifar --method ensemble --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset cifar --method ensemble --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset cifar --method ensemble --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-teacher-20-20220424-223105-seed_1 ./cifar/cifar-teacher-20-20220430-183155-seed_2 ./cifar/cifar-teacher-20-20220430-183242-seed_3 --label cifar-teacher

#### Dropout ####
python3 train.py --gpu 3 --dataset cifar --method drop  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset cifar --method drop  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset cifar --method drop  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-drop-20-20220507-072223-seed_1 ./cifar/cifar-drop-20-20220507-114027-seed_2 ./cifar/cifar-drop-20-20220507-160013-seed_3 --label cifar-drop

#### Gaussian ####
python3 train.py --gpu 3 --dataset cifar --method gauss  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset cifar --method gauss  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset cifar --method gauss  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-gauss-20-20220509-182209-seed_1 ./cifar/cifar-gauss-20-20220509-224525-seed_2 ./cifar/cifar-gauss-20-20220510-031020-seed_3 --label cifar-gauss

#### Hydra ####
python3 train.py --gpu 3 --dataset cifar --method hydra  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset cifar --method hydra  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset cifar --method hydra  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-hydra-20-20220508-165742-seed_1 ./cifar/cifar-hydra-20-20220508-230630-seed_2 ./cifar/cifar-hydra-20-20220509-051506-seed_3 --label cifar-hydra

#### EnDD ####
python3 train.py --gpu 3 --dataset cifar --method endd  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset cifar --method endd  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset cifar --method endd  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-endd-20-20220506-220312-seed_1 ./cifar/cifar-endd-20-20220507-035716-seed_2  ./cifar/cifar-endd-20-20220507-094904-seed_3 --label cifar-endd

#### Hydra + ####
##### 20 tails #####
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-hydra+-20-20220506-185325-seed_1 ./cifar/cifar-hydra+-20-20220506-185319-seed_2  ./cifar/cifar-hydra+-20-20220507-010939-seed_3  --label cifar-hydra+-20

##### 10 tails #####
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 1 --label seed_1
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 2 --label seed_2
python3 train.py --gpu 0 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-hydra+-10-20220509-202337-seed_1 ./cifar/cifar-hydra+-10-20220510-033241-seed_2 ./cifar/cifar-hydra+-10-20220510-104046-seed_3 --label cifar-hydra+-10

##### 5 tails #####
python3 train.py --gpu 1 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220424-223105-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 1 --label seed_1
python3 train.py --gpu 1 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183155-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 2 --label seed_2
python3 train.py --gpu 1 --dataset cifar --method hydra+  --load_teacher ./cifar/cifar-teacher-20-20220430-183242-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./cifar/cifar-hydra+-5-20220509-202436-seed_1 ./cifar/cifar-hydra+-5-20220510-025500-seed_2 ./cifar/cifar-hydra+-5-20220510-092356-seed_3 --label cifar-hydra+-5

python3 compare_results.py --folder_paths ./cifar/cifar-teacher-20220506-144600 ./cifar/cifar-gauss-20220510-101930 ./cifar/cifar-drop-20220508-100356 ./cifar/cifar-endd-20220510-160221 ./cifar/cifar-hydra-20220509-113427 ./cifar/cifar-hydra+-20-20220509-142106 ./cifar/cifar-hydra+-10-20220510-174707 ./cifar/cifar-hydra+-5-20220510-162413 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --label comparison --dataset cifar
python3 compare_outputs.py --folder_paths ./cifar/cifar-teacher-20-20220424-223105-seed_1 ./cifar/cifar-gauss-20-20220509-182209-seed_1 ./cifar/cifar-drop-20-20220507-072223-seed_1 ./cifar/cifar-endd-20-20220506-220312-seed_1 ./cifar/cifar-hydra-20-20220508-165742-seed_1 ./cifar/cifar-hydra+-20-20220508-103702-seed_1 ./cifar/cifar-hydra+-10-20220509-202337-seed_1 ./cifar/cifar-hydra+-5-20220509-202436-seed_1 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --label cifar-seed-1 --dataset cifar
python3 compare_outputs.py --folder_paths ./cifar/cifar-teacher-20-20220430-183155-seed_2 ./cifar/cifar-gauss-20-20220509-224525-seed_2 ./cifar/cifar-drop-20-20220507-114027-seed_2 ./cifar/cifar-endd-20-20220507-035716-seed_2 ./cifar/cifar-hydra-20-20220508-230630-seed_2 ./cifar/cifar-hydra+-20-20220508-194829-seed_2 ./cifar/cifar-hydra+-10-20220510-033241-seed_2 ./cifar/cifar-hydra+-5-20220510-025500-seed_2 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --label cifar-seed-2 --dataset cifar
python3 compare_outputs.py --folder_paths ./cifar/cifar-teacher-20-20220430-183242-seed_3 ./cifar/cifar-gauss-20-20220510-031020-seed_3 ./cifar/cifar-drop-20-20220507-160013-seed_3 ./cifar/cifar-endd-20-20220507-094904-seed_3 ./cifar/cifar-hydra-20-20220509-051506-seed_3 ./cifar/cifar-hydra+-20-20220509-045926-seed_3 ./cifar/cifar-hydra+-10-20220510-104046-seed_3  ./cifar/cifar-hydra+-5-20220510-092356-seed_3 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --label cifar-seed-3 --dataset cifar

### SVHN ###
#### Ensemble ####
python3 train.py --gpu 3 --dataset svhn --method ensemble --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method ensemble --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method ensemble --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-teacher-20-20220509-095821-seed_1  ./svhn/svhn-teacher-20-20220430-183223-seed_2  ./svhn/svhn-teacher-20-20220502-121709-seed_3 --label svhn-teacher

#### Dropout ####
python3 train.py --gpu 3 --dataset svhn --method drop --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method drop --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method drop --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-drop-20-20220510-165926-seed_1 ./svhn/svhn-drop-20-20220508-122302-seed_2 ./svhn/svhn-drop-20-20220508-123845-seed_3 --label svhn-drop

#### Gaussian ####
python3 train.py --gpu 3 --dataset svhn --method gauss --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method gauss --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method gauss --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-gauss-20-20220510-190658-seed_1 ./svhn/svhn-gauss-20-20220509-140704-seed_2 ./svhn/svhn-gauss-20-20220509-164750-seed_3 --label svhn-gauss

#### Hydra ####
python3 train.py --gpu 3 --dataset svhn --method hydra --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method hydra --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method hydra --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-hydra-20-20220510-212530-seed_1 ./svhn/svhn-hydra-20-20220508-195711-seed_2 ./svhn/svhn-hydra-20-20220508-222435-seed_3 --label svhn-hydra

#### EnDD ####
python3 train.py --gpu 3 --dataset svhn --method endd --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method endd --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method endd --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-endd-20-20220510-235332-seed_1 ./svhn/svhn-endd-20-20220508-143937-seed_2 ./svhn/svhn-endd-20-20220508-151224-seed_3 --label svhn-endd

#### Hydra+ ####
##### 20 tails ##### 
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 1 --label seed_1
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 2 --label seed_2
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 20 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-hydra+-20-20220513-201522-seed_1 ./svhn/svhn-hydra+-20-20220513-233850-seed_2 ./svhn/svhn-hydra+-20-20220514-030226-seed_3   --label svhn-hydra+-20

##### 10 tails #####
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 1 --label seed_1
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 2 --label seed_2
python3 train.py --gpu 2 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 10 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-hydra+-10-20220514-062425-seed_1 ./svhn/svhn-hydra+-10-20220514-091339-seed_2 ./svhn/svhn-hydra+-10-20220514-120050-seed_3   --label svhn-hydra+-10

#### 5 tails #####
python3 train.py --gpu 3 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220509-095821-seed_1/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 1 --label seed_1
python3 train.py --gpu 3 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220430-183223-seed_2/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 2 --label seed_2
python3 train.py --gpu 3 --dataset svhn --method hydra+ --load_teacher ./svhn/svhn-teacher-20-20220502-121709-seed_3/weights.pt  --n_tails_teacher 20 --n_tails 5 --seed 3 --label seed_3
python3 average_results.py --folder_paths ./svhn/svhn-hydra+-5-20220513-201608-seed_1 ./svhn/svhn-hydra+-5-20220513-225623-seed_2 ./svhn/svhn-hydra+-5-20220514-013647-seed_3  --label svhn-hydra+-5

python3 compare_results.py --folder_paths ./svhn/svhn-teacher-20220510-173657 ./svhn/svhn-gauss-20220511-112342 ./svhn/svhn-drop-20220511-112155 ./svhn/svhn-endd-20220511-112257 ./svhn/svhn-hydra-20220511-112502 ./svhn/svhn-hydra+-20-20220515-131844 ./svhn/svhn-hydra+-10-20220515-131730 ./svhn/svhn-hydra+-5-20220515-131644 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --dataset svhn --label comparison
python3 compare_outputs.py --folder_paths ./svhn/svhn-teacher-20-20220509-095821-seed_1 ./svhn/svhn-gauss-20-20220510-190658-seed_1 ./svhn/svhn-drop-20-20220510-165926-seed_1 ./svhn/svhn-endd-20-20220510-235332-seed_1 ./svhn/svhn-hydra-20-20220510-212530-seed_1 ./svhn/svhn-hydra+-20-20220513-201522-seed_1 ./svhn/svhn-hydra+-10-20220514-062425-seed_1 ./svhn/svhn-hydra+-5-20220513-201608-seed_1 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5  --label svhn-seed-1 --dataset svhn
python3 compare_outputs.py --folder_paths ./svhn/svhn-teacher-20-20220430-183223-seed_2 ./svhn/svhn-gauss-20-20220509-140704-seed_2 ./svhn/svhn-drop-20-20220508-122302-seed_2 ./svhn/svhn-endd-20-20220508-143937-seed_2 ./svhn/svhn-hydra-20-20220508-195711-seed_2 ./svhn/svhn-hydra+-20-20220513-233850-seed_2 ./svhn/svhn-hydra+-10-20220514-091339-seed_2 ./svhn/svhn-hydra+-5-20220513-225623-seed_2 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5 --label svhn-seed-2 --dataset svhn
python3 compare_outputs.py --folder_paths ./svhn/svhn-teacher-20-20220502-121709-seed_3 ./svhn/svhn-gauss-20-20220509-164750-seed_3 ./svhn/svhn-drop-20-20220508-123845-seed_3 ./svhn/svhn-endd-20-20220508-151224-seed_3 ./svhn/svhn-hydra-20-20220508-222435-seed_3 ./svhn/svhn-hydra+-20-20220514-030226-seed_3 ./svhn/svhn-hydra+-10-20220514-120050-seed_3 ./svhn/svhn-hydra+-5-20220514-013647-seed_3 --labels Ensemble Gauss Drop EnDD Hydra-20 Hydra+-20 Hydra+-10 Hydra+-5  --label svhn-seed-3 --dataset svhn