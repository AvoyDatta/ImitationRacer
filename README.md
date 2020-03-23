# ImitationRacer: Sequential Behavioral Cloning for Autonomous Control
![](vid.gif)

This is our course project for the course CS234: Reinforcement Learning at Stanford University. Our report can be found at:

## Requirements
gym 0.7.4
tensorflow 1.7.1
pyglet 1.3.2

## Various Versions used:

*baseline*: CNN + reduce acceleration

*lstm_0.0.0*: LSTM, sample_interval = 1, no class balancing + reduce accelerate, history_len = 3 

*lstm_0.1.0*: LSTM, sample_interval=1, class balancing, history_len = 3

*lstm_0.0.1*: LSTM, sample_interval=1, no class balancing + reduce accelerate, hist len = 10


*lstm_1.0*: LSTM, sample_interval = 2, no class balancing, history_len = 3 

*lstm_1.0.1*: LSTM, sample_interval = 2,  no class balancing + reduce accelerate, history_len = 5

*Sample train (from ImitationRacer/src)*

python train_agent.py --user <user name> --model <version name> 

*Sample test (from ImitationRacer/src)*

python test_agent.py --user <user name> --model <version name> --ts <time stamp of training (from ckpts/user/model/)> --eps <num episodes>

## Train 
```
python train_agent.py --user <user name> --model <model name> --hist_len <history length> --lr <learning rate> --batch_size <batch size>
```

## Test
```
python test_agent.py --user <user name> --model <model name> --ts <time stamp> --hist_len <history length> 
```

### Acknowledgements
We built our code base on top of Gui Miotto's code base, found here: 
https://github.com/gui-miotto/DeepLearningLab/tree/master/Assignment%2003

### Contact
Please reach out to Avoy (avoy.datta@stanford.edu) or Aditya (adusi@stanford.edu) with questions, comments or feedback.