# ImitationRacer

## Versions:

*baseline*: CNN + reduce acceleration

*lstm_0.0.0*: LSTM, sample_interval = 1, no class balancing + reduce accelerate, history_len = 3 
*lstm_0.1.0*: LSTM, sample_interval=1, class balancing, history_len = 3

*lstm_0.0.1*: LSTM, sample_interval=1, no class balancing + reduce accelerate, hist len = 10


*lstm_1.0*: LSTM, sample_interval = 2, no class balancing, history_len = 3 


*Sample train (from ImitationRacer/src)*

python train_agent.py --user <user name> --model <version name> 

*Sample test (from ImitationRacer/src)*

python test_agent.py --user <user name> --model <version name> --ts <time stamp of training (from ckpts/user/model/)> --eps <num episodes>