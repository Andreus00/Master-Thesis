# train baseline model on user data
echo "Training baseline model on user data"

# echo "Training 122"
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00122/Inner/Take2/
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00122/Outer/Take9/

# echo "Training 123"
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00123/Inner/Take2/
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00123/Outer/Take8/

# echo "Training 127"
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00127/Inner/Take2/
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00127/Outer/Take11/

# echo "Training 147"
# python3 train_baseline_on_user.py --source_path 4D-DRESS/00147/Inner/Take2/
python3 train_baseline_on_user.py --source_path 4D-DRESS/00147/Outer/Take13/

# run retarget from all users to all users

# 122 -> 122
# echo "Running 122 -> 122"
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Outer --src_take Take9 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 #--run_nicp
# python3 video_from_renders.py --src_user 00122 --src_outfit Outer --src_take Take9 --trg_user 00122 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2

# # 123 -> 123
# echo "Running 123 -> 123"
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 --run_nicp
# python3 video_from_renders.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00123 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2

# # 127 -> 127
# echo "Running 127 -> 127"
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 --run_nicp
# python3 video_from_renders.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00127 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2

# # 147 -> 147
# echo "Running 147 -> 147"
python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 --run_nicp
# python3 video_from_renders.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00147 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2


# # 122 -> 147
# echo "Running 122 -> 147"
python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Outer --src_take Take9 --trg_user 00147 --trg_outfit Outer --trg_take Take13 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00122 --src_outfit Outer --src_take Take9 --trg_user 00147 --trg_outfit Outer --trg_take Take13

# # 122 -> 127
# echo "Running 122 -> 127"
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Outer --trg_take Take11 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Outer --trg_take Take11

# # 122 -> 123
# echo "Running 122 -> 123"
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 --run_nicp
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Outer --trg_take Take8 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00122 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Outer --trg_take Take8


# # 123 -> 122
# echo "Running 123 -> 122"
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 --run_nicp
# python3 video_from_renders.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00122 --trg_outfit Inner --trg_take Take2


# # 123 -> 127
# echo "Running 123 -> 127"
python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2
python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00127 --trg_outfit Inner --trg_take Take2

# # 123 -> 147
# echo "Running 123 -> 147"
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00123 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00147 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00147 --trg_outfit Outer --trg_take Take13 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5 
# python3 video_from_renders.py --src_user 00123 --src_outfit Outer --src_take Take8 --trg_user 00147 --trg_outfit Outer --trg_take Take13

# # 127 -> 122
# echo "Running 127 -> 122"
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00122 --trg_outfit Inner --trg_take Take2

# # 127 -> 123
# echo "Running 127 -> 123"
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00123 --trg_outfit Inner --trg_take Take2

# # 127 -> 147
# echo "Running 127 -> 147"
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Inner --src_take Take2 --trg_user 00147 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00147 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00127 --src_outfit Outer --src_take Take11 --trg_user 00147 --trg_outfit Inner --trg_take Take2


# # 147 -> 122
# echo "Running 147 -> 122"
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00122 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00122 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00122 --trg_outfit Inner --trg_take Take2

# # 147 -> 123
# echo "Running 147 -> 123"
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00123 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00123 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00123 --trg_outfit Inner --trg_take Take2

# # 147 -> 127
# echo "Running 147 -> 127"
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Inner --src_take Take2 --trg_user 00127 --trg_outfit Inner --trg_take Take2
# python3 run_retarget_lbs_on_user.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00127 --trg_outfit Inner --trg_take Take2 --src_clothes_label_ids 2 3 4 5 --trg_clothes_label_ids 2 3 4 5
# python3 video_from_renders.py --src_user 00147 --src_outfit Outer --src_take Take13 --trg_user 00127 --trg_outfit Inner --trg_take Take2

