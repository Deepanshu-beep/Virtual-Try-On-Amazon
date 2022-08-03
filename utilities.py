import gdown
import numpy as np
from PIL import Image
import IPython
import gdown
import os
import sys
import u2net_load
import time
import subprocess
import u2net_run
u2net = u2net_load.model(model_name = 'u2netp')

from predict_pose import generate_pose_keypoints


def get_tryon(person, attire):
    cloth_name = attire
    cloth_path = os.path.join('inputs/cloth', sorted(os.listdir('inputs/cloth'))[3])
    cloth = Image.open(cloth_path)
    cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
    cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))

    u2net_run.infer(u2net, 'Data_preprocessing/test_color', 'Data_preprocessing/test_edge')
    start_time = time.time()
    img_name = '000001_0.png'
    img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[0])
    img = Image.open(img_path)
    img = img.resize((192,256), Image.BICUBIC)

    img_path = os.path.join('Data_preprocessing/test_img', img_name)
    img.save(img_path)
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time-start_time))

    # !python3 Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'lip_final.pth' --input-dir 'Data_preprocessing/test_img' --output-dir 'Data_preprocessing/test_label'
    parse_time = time.time()
    print('Parsing generated in {}s'.format(parse_time-resize_time))

    pose_path = os.path.join('Data_preprocessing/test_pose', img_name.replace('.png', '_keypoints.json'))
    generate_pose_keypoints(img_path, pose_path)
    pose_time = time.time()
    print('Pose map generated in {}s'.format(pose_time-parse_time))
    with open('Data_preprocessing/test_pairs.txt','w') as f:
        f.write('000001_0.png 000001_1.png')
    subprocess.run('python','test.py')
    output_grid = np.concatenate([np.array(Image.open('Data_preprocessing/test_img/000001_0.png')),
                np.array(Image.open('Data_preprocessing/test_color/000001_1.png')),
                np.array(Image.open('results/test/try-on/000001_0.png'))], axis=1)
    image_grid = Image.fromarray(output_grid)
    return image_grid