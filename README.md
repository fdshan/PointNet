# PointNet
PyTorch ver.
___
## Classification
with feature transform: \
`python train_classification.py --dataset=../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=5 --feature_transform`

without feature transform: \
`python train_classification.py --dataset=../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=5`
___
## Segmentation

Class `Chair`, with feature transform: \
`python train_segmentation.py --dataset=../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=10 --feature_transform --class_choice='Chair'`

Class `Chair`, without feature transform: \
`python train_segmentation.py --dataset=../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=10 --class_choice='Chair'`
___
## Visualization:

For segmentation:
`python show_seg.py --class_choice='Chair' --dataset=../shapenetcore_partanno_segmentation_benchmark_v0 --model=seg/weights_without_transform/seg_model_Chair_4.pth`
