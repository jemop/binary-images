from utils.imageutils import img_to_binary, img_to_3colors

if __name__=="__main__":

    name="girl_dancing.jpg"

    #processed_2colors_path = img_to_binary(name, filter_size=21, threshold=200)
    processed_3colors_path = img_to_3colors(name, filter_size=21, threshold1=100, threshold2=200)
    print("Processed images succesfully.")