from rmb_lib.video_class import Video_3D

info = ['v_ApplyEyeMakeup_g01_c01', '/data4/zhouhao/dataset/ucf101/tvl1/v_ApplyEyeMakeup_g01_c01', '163','0']
video = Video_3D(info, tag='flow')
print(video)

print(video.get_frames(163).shape)