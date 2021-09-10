set -ex
# models
RESULTS_DIR='./results/discover60k'
G_PATH='../checkpoints/discover60k/discover60k_bicycle_gan/latest_net_G.pth'
E_PATH='../checkpoints/discover60k/discover60k_bicycle_gan/latest_net_E.pth'

# dataset
CLASS='discover60k'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
NUM_TEST=100 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ../checkpoints \
  --name ${CLASS}/${CLASS}_bicycle_gan \
  --dataset_mode ${CLASS} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip \
  --phase test