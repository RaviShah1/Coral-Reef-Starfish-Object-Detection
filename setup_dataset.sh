python cross_validation_setup.py \
    -in ../input/tensorflow-great-barrier-reef/train.csv \
    -out train.csv \
    -type gkfold \
    -on sequence \
    -folds 10 \
    -hold 0

mkdir dataset

python generate_tfrecords.py \
    -c train.csv \
    -o dataset/cots_train \
    -i ../input/tensorflow-great-barrier-reef/train_images \
    -t train \
    -s 4 \
    -f 0

python generate_tfrecords.py \
    -c train.csv \
    -o dataset/cots_val \
    -i ../input/tensorflow-great-barrier-reef/train_images \
    -t valid \
    -s 4 \
    -f 0
