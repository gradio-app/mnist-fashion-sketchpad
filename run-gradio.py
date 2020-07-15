import tensorflow as tf
import gradio as gr
from gradio.inputs import Sketchpad
from gradio.outputs import Label

mnist_model = tf.keras.models.load_model('mnist-fashion-model.h5')

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(inp):
	prediction = mnist_model.predict(inp.reshape(1, 28, 28, 1)).tolist()[0]
	return {class_names[i]: prediction[i] for i in range(10)}


sketchpad = Sketchpad()
label = Label(num_top_classes=4)

gr.Interface(
	predict, 
	sketchpad,  # could also be 'sketchpad' 
	label,
	title="MNIST Fashion Sketch Pad",
        description="This model is trained on the widely known MNIST fashion dataset. Draw an article of clothing and see if the model can guess what it is",
        thumbnail="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset-1024x768.png",
	capture_session=True,
	live=True).launch();
