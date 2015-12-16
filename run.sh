#!/bin/bash -e
#
# Train models and generate predictions.
#

function prep() {
    for file in imgs.zip train.csv w_7489.jpg sample_submission.csv
    do
        if [ ! -f $data_dir/$file ]
        then
            echo $data_dir/$file not found
            exit
        fi
    done
    echo Unzipping: `date`
    unzip -qu $data_dir/imgs.zip -d $data_dir
    cp $data_dir/w_7489.jpg $data_dir/imgs
    mkdir -p $data_dir/train $data_dir/test
    for file in `cat $data_dir/train.csv | cut -f1 -d',' | tail -n +2`
    do
        mv $data_dir/imgs/$file $data_dir/train/
    done
    mv $data_dir/imgs/* $data_dir/test

    echo Cropping training images: `date`
    python crop.py points1.json points2.json $data_dir/train $data_dir/traincrops $imwidth 0

    echo Writing macrobatches: `date`
    python batch_writer.py --image_dir=$data_dir/train --data_dir=$data_dir/macrotrain --points1_file points1.json --points2_file points2.json --target_size $imwidth --val_pct 0
    python batch_writer.py --image_dir=$data_dir/test --data_dir=$data_dir/macrotest --target_size $imwidth --val_pct 100
    python batch_writer.py --image_dir=$data_dir/traincrops --data_dir=$data_dir/macrotraincrops --id_label 1 --target_size $imwidth --val_pct 0

    touch $data_dir/prepdone
    echo Prep done: `date`
}

if [ "$1" == "" ]
then
    echo Usage:  $0 /path/to/data
    exit
fi

data_dir=$1
num_epochs=40
imwidth=384

echo Starting: `date`
echo data_dir=$data_dir, num_epochs=$num_epochs, imwidth=$imwidth

if [ -f $data_dir/prepdone ]
then
    echo $data_dir/prepdone exists. Skipping prep...
else
    prep
fi

echo Localizing first point: `date`
./localizer.py -z32 -e $num_epochs -w $data_dir/macrotrain -tw $data_dir/macrotest -r0 -s model1.pkl -bgpu -pn 1 -iw $imwidth --serialize 1 ${@:2}
echo Localizing second point: `date`
./localizer.py -z32 -e $num_epochs -w $data_dir/macrotrain -tw $data_dir/macrotest -r0 -s model2.pkl -bgpu -pn 2 -iw $imwidth --serialize 1 ${@:2}

echo Cropping test images: `date`
python crop.py testpoints1.json testpoints2.json $data_dir/test $data_dir/testcrops $imwidth 1

echo Writing macrobatches: `date`
python batch_writer.py --image_dir=$data_dir/testcrops --data_dir=$data_dir/macrotestcrops --id_label 1 --target_size $imwidth --val_pct 100

echo Classifying: `date`
./classifier.py -z32 -e 60 -w $data_dir/macrotraincrops -tw $data_dir/macrotestcrops -r0 -s model3.pkl -bgpu -iw $imwidth --serialize 1 ${@:2}
echo Done: `date`
