# ASL Real-Time detection and classification

## Get it working
1. Firstly, just run this command to create virtual enviroment with all requirements:
```
conda create --name <env> --file requirements.txt
```
and activate it:
```
conda activate <env>
```
2. Download weights for models from [here](https://drive.google.com/drive/folders/1GQLPHOC0Dbp3iny4N_-8Q5mWPorvwN5K?usp=sharing) into the folder `weights` (you should create it).

3. Then run (on `linux` use `python3` instead of `python`)
```
python main.py
```
Alternatively, you can start `jupyter notebook` and open file `main.ipynb` (and run cells, of course).
<br>To exit the program just press `q` when its window is active.

## Get your hands dirty
### Detection
The original code is [here](https://github.com/zllrunning/hand-detection.PyTorch). I used it for simplifying my work. However, I had to make some changes to make that code work.
1. Firstly, you need to get data. Being in folder `detection` run the following command and the EgoHands dataset will be downloaded and preprocessed.
```
python prepare_data.py
```
*(Remember about `python3` and `linux`)*<br>
2. When you get data just run the following command to train the FaceBoxes model for hand detection.
```
python ../train_detection.py
```
The arguments you can specify for it:
* --training_dataset, default=./data/Hand, Training dataset directory
* -b, --batch_size, default=20, Batch size for training
* --num_workers, default=8, Number of workers used in dataloading
* --ngpu, default=0, Number of GPU's, usually, there is 1 (0 when train on CPU)
* --lr, --learning-rate, default=1e-3, Initial learning rate
* --momentum, default=0.9, Momentum
* --resume_net, default=None, Specify path for weights if you want to continue training
* --resume_epoch, default=0, Resume iter for retraining
* -max, --max_epoch, default=50, Number of epochs for training
* --weight_decay, default=5e-4, Weight decay for SGD
* --gamma, default=0.1, Gamma update for SGD
* --save_folder, default=weights, Location to save weights (and checkpoint weights); remember that by default main program takes weights for models from ../weights/

### Classification
1. Get dataset [here](https://www.kaggle.com/grassknoted/asl-alphabet#E1005.jpg).
2. Run the following to reorganize the dataset a bit.
```
python asl_preproccess.py
```
3. Open `classification.ipynb` and train GoogLeNet model with preferable parameters. Since it's already pre trained on pretty big dataset, the small number of epochs is neede to teach it to classify ASL letters.