import os
from PIL import Image

def main() :

    print(" step 0: save dir")
    save_dir = "/home/dreamyou070/MyData/ADE_preprocessed"
    os.makedirs(save_dir, exist_ok=True)

    print(" step 1: base dir")
    base_dir = "/home/dreamyou070/MyData/ADE/ADE20K_2021_17_01/images/ADE"
    test_conditions = os.listdir(base_dir)
    for test_condition in test_conditions:
        test_condition_dir = os.path.join(base_dir, test_condition)
        categories = os.listdir(test_condition_dir)
        save_test_condition_dir = os.path.join(save_dir, test_condition)
        os.makedirs(save_test_condition_dir, exist_ok=True)

        for category in categories:
            category_dir = os.path.join(test_condition_dir, category)
            sub_categories = os.listdir(category_dir)
            save_category_dir = os.path.join(save_test_condition_dir, category)
            os.makedirs(save_category_dir, exist_ok=True)
            image_base_dir = os.path.join(save_category_dir, "images")
            mask_base_dir = os.path.join(save_category_dir, "masks")
            os.makedirs(image_base_dir, exist_ok=True)
            os.makedirs(mask_base_dir, exist_ok=True)

            for sub_category in sub_categories:
                sub_category_dir = os.path.join(category_dir, sub_category)
                files = os.listdir(sub_category_dir)
                for file in files:
                    name, ext = os.path.splitext(file)
                    if ext == ".jpg":
                        img_dir = os.path.join(sub_category_dir, file)
                        mask_dir = os.path.join(sub_category_dir, name + "_seg.png")

                        save_img_dir = os.path.join(image_base_dir, file)
                        Image.open(img_dir).resize((512, 512)).save(save_img_dir)

                        save_mask_dir = os.path.join(mask_base_dir, name + file)
                        Image.open(mask_dir).resize((256,256)).save(save_mask_dir)

                        #name_folder = os.path.join(mask_base_dir, name)
                        json_file = os.path.join(save_category_dir, f'{name}.json')
                        




if __name__ == '__main__':
    main()