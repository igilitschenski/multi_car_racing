# Instructions to train different agents
## A3C
Simply run
```
python a3c.py
```
More options coming soon.

## DDPG
cd into
``` 
../main_scripts 
```
and run 
``` 
python train_test_DDPG.py
``` 
and set the desired arguments (see help for explanation). If running on a server: run 
```
xvfb-run -s "-screen 0 1400x900x24" python train_test_DDPG.py
```
