from utils.imageutils import img_to_binary

if __name__=="__main__":

    name="alex.jpg"

    processed_img_path = img_to_binary(name, filter_size=11, threshold=100)
    print("Processed image succesfully. Image saved in "+processed_img_path)