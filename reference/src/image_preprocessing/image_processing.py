import glob
import os.path

from PIL import Image


def crop_square(image):
    width = int(image.size[0])
    height = int(image.size[1])
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    min_dim = int(min(width, height))

    width_crop_val = (width - min_dim) // 2
    height_crop_val = (height - min_dim) // 2
    img2 = image.crop(
        (
            width_crop_val,
            height_crop_val,
            width - width_crop_val,
            height - height_crop_val
        )
    )
    return img2


def resize(image, size):
    """
        Resizes input file
    :param image: Image object of input file
    :param size: size of output file
    :return:   Image object of resized image
    """
    img = None
    try:
        img = image
        img.thumbnail(size)
    except IOError:
        print('cannot read input file')
    return img


def get_file_names(image_dir, extensions=('JPG', 'JPEG', 'jpg', 'jpeg')):
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, "*." + extension)  # '.data/train/*.png'
        file_list.extend(glob.glob(file_glob))  # list of all files with same pattern
    return file_list


def main(verify=False):
    image_dirs = glob.glob('data/images/*')
    if verify is not True:
        for image_dir in image_dirs:
            file_list = get_file_names(image_dir)
            for file_loc in file_list:
                image = Image.open(file_loc)
                image_name = os.path.basename(file_loc)
                image = crop_square(image)
                image = resize(image, (224, 224))
                new_dir = os.path.join(image_dir, 'prepared')
                if not os.path.isdir(new_dir):
                    os.mkdir(new_dir)
                image.save(os.path.join(new_dir, image_name), "JPEG")
    else:
        for dir in image_dirs:
            check_size(dir)


def check_size(dir):
    file_list = get_file_names(dir)
    for file_loc in file_list:
        image = Image.open(file_loc)
        if image.size != (224, 224):
            print(file_loc)


if __name__ == '__main__':
    main(verify=True)
