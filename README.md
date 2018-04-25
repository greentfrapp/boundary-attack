# boundary-attack
Implementation of the Boundary Attack algorithm as described in:

Brendel, Wieland, Jonas Rauber, and Matthias Bethge. **"Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models."** *arXiv preprint arXiv:1712.04248 (2017).*

<div>
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_231253_dingo.png" alt="doge_1" width="100px" height="whatever" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_231254_dingo.png" alt="doge_2" width="100px" height="whatever" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_231307_dingo.png" alt="doge_3" width="100px" height="whatever" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_231453_dingo.png" alt="doge_4" width="100px" height="whatever" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_232244_dingo.png" alt="doge_5" width="100px" height="whatever" style="display: inline-block;">
<img src="https://raw.githubusercontent.com/greentfrapp/boundary-attack/master/images/sample_4_label273_dingo/20180422_234213_dingo.png" alt="doge_6" width="100px" height="whatever" style="display: inline-block;">
</div>

*All of the above images are classified as `273: 'dingo, warrigal, warragal, Canis dingo'` by the Keras ResNet-50 model pretrained on ImageNet*

## Instructions

For starters, just run:

```
$ python boundary-attack-resnet.py
```

This will create adversarial images using the Bad Joke Eel and Awkward Moment Seal images, for attacking the Keras ResNet-50 model (pretrained on ImageNet). You can also change the files to other images in the `images/original` folder or add your own images. All input images will be reshaped to 224 x 224 x3 arrays.

##MNIST Demo

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
