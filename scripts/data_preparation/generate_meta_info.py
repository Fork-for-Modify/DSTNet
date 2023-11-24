from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    # gt_folder = '/media/xbm/data/VideoDeblur_Dataset/BSD/BSD_3ms24ms_2/train/blur/'
    # meta_info_txt = '/media/xbm/data/xbm/BasicSR/BasicSR_wave/basicsr/data/meta_info/meta_info_BSD_GT.txt'
    gt_folder = '/ssd/1/yrz/Dataset/GoPro_INR/GoPro_train_blur_gamma_ratio_5_steps_20000_dec/'
    meta_info_txt = './basicsr/data/meta_info/meta_info_GoPro_crop_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))
    print(img_list)

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_div2k()
