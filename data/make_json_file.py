import os
import json


def main() :

    # [1] json file
    json_file_dir = r'dataset_0.json'
    json_file = open(json_file_dir, 'r')
    json_data = json.load(json_file)

    # [1] base position
    base_dir = ''
    base_img_dir = os.path.join(base_dir, 'imagesTr')
    base_label_dir = os.path.join(base_dir, 'labelsTr')
    files = os.listdir(base_img_dir)
    for file in files:
        name, ext = os.path.splitext(file)
        label_name = name.replace('img', 'label')
        img_path = f'imagesTr/{file}'
        label_path = f'labelsTr/{label_name}{ext}'
        value = {'image': img_path,
                 'label': label_path}
        if value not in json_data['validation'] :
            json_data['validation'].append(value)

    # [3] save json file
    new_json_file_dir = r'dataset_0.json'
    with open(new_json_file_dir, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

if __name__ == '__main__':
    main()
