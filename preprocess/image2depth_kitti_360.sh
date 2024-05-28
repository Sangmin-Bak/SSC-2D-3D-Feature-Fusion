set -e
exeFunc(){
    cd mobilestereonet
    baseline=$1
    num_seq=$2
    CUDA_VISIBLE_DEVICES=0 python prediction.py --datapath ../../kitti-360/data_2d_raw/$num_seq \
    --testlist ./filenames_kitti_360/$num_seq.txt --num_seq $num_seq --loadckpt ./MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti360 \
    --model MSNet3D --savepath "./kitti_360_depth" --baseline $baseline
    cd ..
}

# Change data_path to your own specified path
# And make sure there is enough space under data_path to store the generated data
# data_path=/mnt/NAS/data/yiming/segformer3d_data
data_path=/home/sangmin/workspace/3DSemanticSceneCompletion/VoxFormer/kitti-360/data_2d_depth

sequences="
2013_05_28_drive_0000_sync 
2013_05_28_drive_0002_sync 
2013_05_28_drive_0003_sync 
2013_05_28_drive_0004_sync 
2013_05_28_drive_0005_sync
2013_05_28_drive_0006_sync 
2013_05_28_drive_0007_sync
2013_05_28_drive_0009_sync
2013_05_28_drive_0010_sync"

mkdir -p $data_path
ln -s $data_path ./mobilestereonet/kitti_360_depth
for i in $sequences
do
    exeFunc 388.1823 $i     
done


