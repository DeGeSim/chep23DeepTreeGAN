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
$ python -m fgsim --hash fea87f9 test
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
$ python -m fgsim --tag t_retrain setup
> Experiment setup with hash 8dea68a.
$ python -m fgsim --hash 8dea68a train
>
23-09-20 15:39 INFO   tag: t_retrain hash: 8dea68a loader_hash: 0d09873
           INFO   Running command train
           WARNING  Proceeding without loading checkpoint.
           WARNING  Starting with state epoch: 0
                processed_events: 0
                grad_step: 0
                complete: false
           INFO   Using the first 50 batches for validation and the next 250 batches for testing.
           INFO   Device: Tesla V100-SXM2-32GB
           INFO   Validating
Generating eval batches: 100%|████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.04it/s]
           INFO   Postprocessing
           INFO   Postprocessing done
           INFO   w1m 118.82    w1p 42.76     fpnd 198.90   auc 0.03      w1disc 1.24
           WARNING  New best model at step 0
WARNING:fgsim:New best model at step 0
Epoch 0:   4%|████▏                                                                   | 589/14725 [00:51<20:30, 11.49it/s]
Epoch 1:   8%|████████▎                                                               | 1178/14725 [00:46<17:50, 12.65it/s]
Epoch 2:  12%|████████████▍                                                           | 1767/14725 [00:44<16:08, 13.37it/s]
Epoch 3:  14%|██████████████
~~~
