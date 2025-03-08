# The Atlas Docker Container

This archive contains an example of a Dockerfile that can be used to build images with Python.
In addition to installing Python, it creates a Python virtual environment and installs some Python packages (see requirements.txt). Then, it defines a working directory and copies some files (check_pytorch.py, my_network_infer.py, unet.py, my_network.pt) from the host (your machine) into the image. When the image is executed a bash shell is run


## Documentation

For more information about the challenge and the dataset, you are invited to visit the [Atlas website](https://atlas-challenge.u-bourgogne.fr).


## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

