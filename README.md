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

Retrain the models:
~~~bash
$ bash setup_venv.sh
> Experiment setup with hash 1f62bc2.
$ python -m fgsim --hash 1f62bc2 train
> ...
~~~
