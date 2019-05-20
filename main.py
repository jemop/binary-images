from utils.imageutils import img_to_binary

if __name__=="__main__":

    name="diegolau1.jpeg"

    processed_img_path = img_to_binary(name, filter_size=11, threshold=200)
    print("Processed image succesfully. Image saved in "+processed_img_path)