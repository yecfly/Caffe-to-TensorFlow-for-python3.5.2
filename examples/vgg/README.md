
_Thanks to Visual Geometry Group£¨http://www.robots.ox.ac.uk/~vgg/software/vgg_face/£© for this VGG-Face prototxt example_

The full VGG_FACE caffe model can be downloaded in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

First, you can convert a prototxt model to tensorflow code:

    $ ./convert.py examples/vgg/VGG_FACE_deploy.prototxt --code-output-path=vggface.py

This produces tensorflow code for the VGG_Face network in `vggface.py`.  For more information, please go to mnist example.
