import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from moviepy.editor import VideoFileClip

# Define the path to your AVI file
avi_file_path = r'U:\\MAIN\\HyperCANs\\20251017_MeetingPre\\Sample_2_rotational_damping\\yz_biaxial_rotation_corr.avi'

# Define the output GIF file path
gif_file_path = r'U:\\MAIN\\HyperCANs\\20251017_MeetingPre\\Sample_2_rotational_damping\\yz_biaxial_rotation_corr.gif'

# Load the AVI file
video_clip = VideoFileClip(avi_file_path)

# Convert the video to GIF
video_clip.write_gif(gif_file_path)

# Close the video clip object
video_clip.close()


# import qrcode
# import qrcode.image.svg

# img = qrcode.make('https://arxiv.org/abs/2503.19832', image_factory=qrcode.image.svg.SvgImage)

# with open('qr.svg', 'wb') as qr:
#     img.save(qr)