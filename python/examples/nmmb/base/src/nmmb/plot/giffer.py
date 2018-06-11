from pycompss.api.task import task
from pycompss.api.parameter import *


@task(gif_name=FILE_OUT, varargsType=FILE_IN)
def generate_animation(gif_name, skip_frames0, *args):
    """
    Generates an animation for the given figures
    :param gif_name: Output file
    :param skip_frames0: Skip all frames that end with _0.png
    :param args: List of files to process
    """
    imgs = list(args)
    # list.sort(imgs)
    if skip_frames0:
        # Remove the ones that end up with _0.png from the list
        file_list = [x for x in imgs if not x.endswith('_0.png')]

    image_list = 'image_list.txt'
    with open(image_list, 'w') as imgs_file:
        for item in file_list:
            imgs_file.write("%s\n" % item)

    from subprocess import Popen, PIPE
    p = Popen('convert -delay 20 @image_list.txt ' + gif_name,
              shell=True,
              stdout=PIPE)
    o, e = p.communicate()
