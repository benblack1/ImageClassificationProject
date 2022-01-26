from ast import arg
import iptcinfo3, os, sys
from numpy.lib.function_base import append
import tensorflow as tf
import numpy as np
import PIL.Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
import pytesseract
from nltk.corpus import wordnet as wn
import tkinter as tk
from tkinter import StringVar, ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import threading
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2)

BAD_HYPERNYMS = set(['entity','physical_entity','object','whole','chordate','vertebrate','placental','matter','substance','artifact',
                     'organism','vascular_plant','angiospermous_tree', 'living_thing', 'abstraction', 'instrumentality','information',
                     'reproductive_structure'])

root = tk.Tk()

GREETING_TEXT = """Hello! Welcome to Ben Black's Senior Project, where you can use AI to help you manage your photo libraries. This nifty program will use machine learning to look at each image in a folder of your choosing, then utilize the power of AI to intelligently tag that image with whatever text or objects it recognizes. To begin, click the "Browse" button."""

class Image:
    def __init__(self, path):
        self.path = path
        self.name = path.split('\\')[-1]
        self.predictions = []
        self.certainty = []
        self.text = ""
        self.hypernyms = []

class Images:
    def __init__(self):
        self.images = list()

    def add_images(self, list_of_images):
        for image in list_of_images:
            self.images.append(Image(image))


def get_list_of_images(directory, list_of_images):
    for entry in os.scandir(directory):
        if entry.is_file():
            if entry.path.endswith(('.jpeg','.jpg','.tiff')): # Filter to only the supported file types
                if entry.path.find("._") == -1: # We don't want temp files, which start with ._ -1 is returned if not found
                    list_of_images.append(entry.path) # Append the path of the image to the running list of images in this directory

            elif entry.path.endswith('.png'):
                if entry.path.find("._") == -1: # We don't want temp files, which start with ._ -1 is returned if not found
                    im = PIL.Image.open(entry.path)
                    jpg = im.convert("RGB")
                    jpg.save(entry.path[:-4] + '.jpg')

        elif entry.is_dir():
            get_list_of_images(entry.path, list_of_images)

    return list_of_images


def classify_images(images):
    # Load pre-built image classifier
    # mobile = tf.keras.applications.mobilenet.MobileNet()
    model = load_model("C:\\Users\\ibben\\Files\\School\\University\\SeniorProject\\IterateThroughPics\\model.h5") # We saved the model so we don't have to download it each time

    for pic in images.images:
        # Pre-processing image
        img = image.load_img(pic.path, target_size=(600,600))
        resized_img = image.img_to_array(img)
        final_image = np.expand_dims(resized_img, axis = 0) # Need fourth dimension
        final_image = tf.keras.applications.efficientnet.preprocess_input(final_image)

        # Make predictions for a single image in the list of images
        predictions = model.predict(final_image)
        # Decode predictions
        decoded_predictions = imagenet_utils.decode_predictions(predictions)
        # Append each prediction and certainty separately, in the same order, to the Image object
        for i, prediction in enumerate(decoded_predictions[0]): # Get 0 because the predictions are wrapped in a single entry list
            if prediction[2] > 0.5 or i == 0: # For every prediction that's above 50% certain (if none, get at least the most certain)
                pic.predictions.append(prediction[1]) # The predicted object is the second element in the prediction tuple
                pic.certainty.append(prediction[2]) # The certainty is the the third element


def find_hypernyms_of_predictions(images):
    for image in images.images:
        # Go through all the predictions we kept (1st, or above 50%)
        for i in range(len(image.predictions)):
            # Turn each prediction into a synset object
            word = wn.synsets(image.predictions[i])[0]
            # Get the hypernyms on each word 8 generations back
            image.hypernyms.extend(list([i for i in word.closure(lambda s:s.hypernyms())])[0:8]) # Get hypernyms 8 generations back
        # Remove duplicates if any
        hypernyms = set(image.hypernyms)
        # Get words from synset object, also remove any hypernyms found in the "Don't want" list
        image.hypernyms = list({element.lemmas()[0].name() for element in hypernyms if element.lemmas()[0].name() not in BAD_HYPERNYMS})


def find_text_in_images(images):
    for pic in images.images:
        img = PIL.Image.open(pic.path)
        jpg = img.convert("RGB")
        text = pytesseract.image_to_string(jpg) # Add any text we find
        pic.text = text.replace("\n"," ") # Replace new lines with spaces


def remove_underscores_from_words(images):
    for image in images.images:
        # Replace all underscores from predictions
        for i, pred in enumerate(image.predictions):
            image.predictions[i] = pred.replace("_"," ")
        # Replace all underscores from hypernyms
        for i, hyper in enumerate(image.hypernyms):
            image.hypernyms[i] = hyper.replace("_"," ")


def tag_images(images):
    for image in images.images: # The Images object stores a list of Image objects. Makes it easier to pass around
        pic_info = iptcinfo3.IPTCInfo(image.path, force=True) # Get the metadata for the image
        # Add predictions as tags
        for i in range(len(image.predictions)): 
            try:
                pic_info['keywords'].append(image.predictions[i]) # Append that prediction as a tag
            except OSError as e:
                print(e)
            except TypeError as e:
                print(e)
            except AttributeError as e:
                print(e)
        # Add hypernyms as tags
        for i in range(len(image.hypernyms)): 
            try:
                pic_info['keywords'].append(image.hypernyms[i]) # Append hypernyms as tags
            except OSError as e:
                print(e)
            except TypeError as e:
                print(e)
            except AttributeError as e:
                print(e)
        # Add text as tags
        try:
            pic_info['keywords'].append(image.text) # Append found text as a tag
            pic_info.save() # Save the tags after iterating through the list of predictions
            os.remove(image.path + '~') # Remove the temp file
        except OSError as e:
            print(e)
        except TypeError as e:
            print(e)
        except AttributeError as e:
            print(e)




def gui():
    root.title("Tag images with AI")
    # root.attributes('-topmost', 1)
    # root.iconbitmap("C:\\Users\\ibben\\Files\\School\\University\\SeniorProject\\IterateThroughPics\\ai.ico")

    window_width = int(root.winfo_screenwidth() // 3)
    window_height = int(root.winfo_screenheight() // 3)
    root.geometry(f'{window_width}x{window_height}')

    greeting = ttk.Label(root, text=GREETING_TEXT, wraplength=(window_width - 20))
    greeting.grid(padx=10,pady=10,columnspan=2)
    directory = StringVar()
    directory.set('Selected Folder: No folder selected')
    dir_label = ttk.Label(root, textvariable=directory,wraplength=(window_width - 20))
    dir_label.grid(padx=10,pady=10,columnspan=2)

    def select_folder():
        filename = fd.askdirectory(
            title='Open a folder',
            initialdir='/')

        directory.set(str("Selected Folder: " + filename))
    
    browse = ttk.Button(root, text="Browse", command=select_folder)
    browse.grid(row=2,column=0,padx=10,pady=10,sticky="E")

    output = tk.Text(root, width=(window_width // 9), height=10)
    output.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def do_everything(directory):
        output.insert(tk.END, f"{directory.get()}")
        if(directory.get() == "Selected Folder: No folder selected" or directory.get() == "Selected Folder: "):
            output.insert(tk.END, "\nInvalid folder. Please try again\n")
            return
        browse["state"] = "disabled"
        begin["state"] = "disabled"
        begin["text"] = "Running"
        output.insert(tk.END, "\nStarting...")
        images = Images() # Images object to hold our list of images
        output.insert(tk.END, "\nGetting all images...")
        list_of_files = []
        list_of_images = get_list_of_images(directory.get()[17:], list_of_files) 
        output.insert(tk.END, f"\nFound {len(list_of_images)} images.")
        images.add_images(list_of_images) # Add those file-paths to the images object. For each file, the Images object
                                        # creates a new Image with the file path and name
        output.insert(tk.END, "\nClassifying images...")
        classify_images(images)
        output.insert(tk.END, "\nFinished classifying. Starting hypernym lookup...")
        find_hypernyms_of_predictions(images)
        output.insert(tk.END, "\nFinished hypernym lookup. Looking for text in images...")
        find_text_in_images(images)
        output.insert(tk.END, "\nFinished finding text. Formatting...")
        remove_underscores_from_words(images)
        output.insert(tk.END, "\nSaving images...")
        tag_images(images)
        output.insert(tk.END, "\nFinished!")
        begin["state"] = "enabled"
        browse["state"] = "enabled"
        begin["text"] = "Start"

    begin = ttk.Button(root, text='Start', command=lambda: threading.Thread(target=do_everything, args=(directory,)).start())
    begin.grid(row=2,column=1,padx=10,pady=10,sticky="W")


    root.mainloop()


if __name__ == '__main__':
    gui()

