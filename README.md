# CANSSI-Ontario-French-Trot-Horse-Racing-Forecasting-Competition
This is the model used to predict the French Trot Horse Racing, 
which is to predict the wining probability.
We use deep-learning algorithms to construct a regression neural network model.

## Introduction for different files
All files that are related with the regression model are stored into the model folder.
I am going to provide a brief introduction for these files.
### Neural_network.py
This is our main file. It's jobs include: load the data from csv file, 
making data pre-process, transform data into Dataset and load it by Dataloader.
Then we create our regression neural network model, which has 4 linear layers.

which involves transforming categorical variables into
numerical types, cleaning all missing values and then restructure data into tensors.
The variables we choose to train our model are:'Barrier', 'Distance', 'IntGender', 
'HandicapDistance','HindShoes', 'HorseAge', 'IntRacingSubType','IntStartType', 
'StartingLine', 'IntSurface', which are our features. Our label is 'FinishPosition'.

### eval_model.py
This file is used to evaluate the regression model.
