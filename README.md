Setup the enviroment:

~~~bash
$ bash setup_venv.sh
$ source venv/bin/activate
~~~

Run the testing:
~~~bash
$ bash setup_venv.sh
$ python -m fgsim --hash fea87f9 train # top quarks
$ python -m fgsim --hash a6e035a train # light quarks
$ python -m fgsim --hash 3d60891 train # gluons
~~~

~~~
❯python -m fgsim --hash fea87f9 test
23-09-20 14:38 INFO   tag: uc_t_dmp hash: fea87f9 loader_hash: 860c256
         INFO   Running command test
         WARNING  Loaded model from checkpoint at epoch 10014 grad_step 5758045.
         WARNING  Starting with state epoch: 10014
            processed_events: 1151610000
            grad_step: 5758045
            complete: true
            best_step: 4044000
            best_epoch: 7033
            time_train_step_start: 1678378023.0828917
            time_io_end: 1678378022.9928455
            time_train_step_end: 1678378023.0827272

         INFO   Loading test dataset from wd/uc_t_dmp/fea87f9/test_best/testdata.pt
         INFO   Evalutating best dataset
23-09-20 14:39 INFO   Metric w1efp took 58.904263 sec
         INFO   Metric fpnd took 18.531814 sec
         INFO   {'w1m': (0.621227669864893, 0.09321568783334482), 'w1p': (1.154803651010956, 0.487636156336616), 'w1efp':
            (1.5358753756792711, 0.3287367789723746), 'fpnd': (0.13744711100389395, nan)}
~~~


Retrain the models:
~~~bash
$ bash setup_venv.sh
> Experiment setup with hash 1f62bc2.
$ python -m fgsim --hash 1f62bc2 train
> ...
~~~
