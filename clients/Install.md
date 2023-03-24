Installation instructions
-------------------------

This guide assumes you are working under Ubuntu 16.04;
however, it can be straightforwardly adapted to any Unix-like system.

1. Clone this repository into some folder

        cd ~; mkdir tmp; cd tmp
        git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients.git

2. Make sure you have Python >= 3.5.3 on your system. If that is not the case,
   install Python3.6

        sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install python3.6
        sudo apt-get install python3.6-venv
    
    More details on that can be found [here](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get).

3. Create a virtual environment, activate it, and update it.
   You can also use an Anaconda virtual environment.

        python3.6 -m venv venv3
        source venv3/bin/activate
        pip3 install -U pip setuptools

4. Install the `quanser_robots` package

        cd clients
        pip3 install -e .

5. Check that everything works correctly by running the code snippet
   from the [main readme file](../Readme.md).
