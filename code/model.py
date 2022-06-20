from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import tensorflow as tf

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
#build_vgg19_unet
def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    # vgg19.trainable=False
    '''S1 skip connection params'''
    vgg19.layers[1].trainable=False                     ## S1 block1_conv1
    vgg19.layers[2].trainable=False                     ## S1 block1_conv2

    '''S2 skip connection params'''
    vgg19.layers[4].trainable=False                     ## S2 block2_conv1
    vgg19.layers[5].trainable=False                     ## S2 block2_conv2

    # '''S3 skip connection params'''
    # vgg19.layers[7].trainable=False                     ## S3 block3_conv1
    # vgg19.layers[8].trainable=False                     ## S3 block3_conv2
    # vgg19.layers[9].trainable=False                     ## S3 block3_conv3
    # vgg19.layers[10].trainable=False                    ## S3 block3_conv4

    # '''S4 skip connection params'''
    # vgg19.layers[12].trainable=False                    ## S4 block4_conv4
    # vgg19.layers[13].trainable=False                    ## S4 block4_conv4
    # vgg19.layers[14].trainable=False                    ## S4 block4_conv4
    # vgg19.layers[15].trainable=False                    ## S4 block4_conv4

    # '''Bridge layer params'''
    # vgg19.layers[17].trainable=False                    ## B1 block5_conv4
    # vgg19.layers[18].trainable=False                    ## B1 block5_conv4
    # vgg19.layers[19].trainable=False                    ## B1 block5_conv4
    # vgg19.layers[20].trainable=False                    ## B1 block5_conv4
    
    """ Encoder """
    # vgg19.get_layer("block1_conv2").trainable = False
    # vgg19.get_layer("block2_conv2").trainable = False
    # vgg19.get_layer("block3_conv4").trainable = False
    # vgg19.get_layer("block4_conv4").trainable = False
    # vgg19.get_layer("block5_conv4").trainable = False
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    # s1.trainable=False
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    # s2.trainable=False
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    # s3.trainable=False
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)
    # s4.trainable=False

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)


    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_vgg19_unet(input_shape)
    model.summary()
