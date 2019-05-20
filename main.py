from utils.imageutils import img_to_binary

if __name__=="__main__":

    name="girl_dancing.jpg"

    processed_img_path = img_to_binary(name, filter_size=21, threshold=200)
    print("Processed image succesfully. Image saved in "+processed_img_path)