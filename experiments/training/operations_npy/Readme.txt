Lanzamiento de los modelos:
python create_mirror.py ./user_14058/poses_full.npy ./user_14058/poses_full_mirror.npy
python rotate_X.py ./user_14058/poses_full.npy ./user_14058/poses_full_rotated_180.npy 180
python Scale.py ./user_14058/poses_full.npy ./user_14058/poses_full_scaled.npy 120
python Shift.py ./user_14058/poses_full.npy ./user_14058/poses_shifted_05_-03.npy 0.5 -0.3
python Shift.py ./user_14058/poses_full.npy ./user_14058/poses_shifted_-025_-03.npy -0.25 -0.3
python Shift.py ./user_14058/poses_full.npy ./user_14058/poses_shifted_-025_03.npy -0.25 0.3
python micro_noise.py ./user_14058/poses_full.npy ./user_14058/poses_noisy.npy 0.005 0.005