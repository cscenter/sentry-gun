__author__ = 'vasdommes'

import subprocess
import os

if __name__ == '__main__':
    # get media info
    input_path = os.path.join("input", "input_HD.avi")
    media_info = subprocess.check_output(
        ["mediainfo.exe", input_path,
         "--Inform=Video;%FrameCount% %Width% %Height% %FrameRate%"])
    print(media_info)
    media_info = media_info.split()
    frames_count = int(media_info[0])
    width = int(media_info[1])
    height = int(media_info[2])
    fps = float(media_info[3])
