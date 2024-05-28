import os

sequences = ['2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
             '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
             '2013_05_28_drive_0007_sync', '2013_05_28_drive_0006_sync', '2013_05_28_drive_0009_sync']

root = './preprocess/mobilestereonet'
data_path = './kitti-360/data_2d_raw'
save_path = os.path.join(root, 'filenames_kitti_360')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for sequence in sequences:
    
    left_path = os.path.join(data_path, sequence, 'image_00', 'data_rect')
    right_path = os.path.join(data_path, sequence, 'image_01', 'data_rect')
    
    left_file_names = sorted(os.listdir(left_path))
    right_file_names = sorted(os.listdir(right_path))
    
    left_file_names = [os.path.join('image_00/data_rect', name) for name in left_file_names]
    right_file_names = [os.path.join('image_01/data_rect', name) for name in right_file_names]
    
    save_file_name = sequence + '.txt'
    with open(os.path.join(save_path, save_file_name), 'w') as f:
        for left_name, right_name in zip(left_file_names, right_file_names):
            f.write(left_name + ' ' + right_name + '\n')
            
    print(f'{save_file_name} is saved...!')
        
    
    