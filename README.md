# boundary-attack
Implementation of the Boundary Attack algorithm as described in:

Brendel, Wieland, Jonas Rauber, and Matthias Bethge. **"Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models."** *arXiv preprint arXiv:1712.04248 (2017).*

For starters, just run:

```
$ python boundary-attack-resnet.py
```

This will create adversarial images using the Bad Joke Eel and Awkward Moment Seal images, for attacking a ResNet-50 model (Keras pre-trained model). You can also change the files to other images in the `images/original` folder or add your own images. The `preprocess` function will use Keras preprocessing functions to reshape an input image to a 224 x 224 x3 array.

There is also a GUI demo (uses Python3) for MNIST images, using a local convolutional model that follows the architecture described [here](https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier), which achieved 98% accuracy on the MNIST test set.

To run the demo, run the following commands:

```
$ cd demo
$ open index.html
$ python3 -m server
```
Wait for the following message to appear:

```
* Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)
```

Then enter the labels for the attack and target and hit Enter. Wait for the MNIST images to load then click on the Attack! button.
