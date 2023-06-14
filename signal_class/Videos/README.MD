To create signal based on an intensite value of a pixel from a video you should:
- upload a video to this dir
- run the function 
  ```python
  dump_all_intensities_of_all_pixels_in_video(video_path, npy_path)
  ``` 
   and create the npy file

- and lastly, to give the signal generator the paths of the video file and the npy you created.
  ```python
    video_path = "signal_class/Videos/Treadmill_Running_500fps.avi"
    npy_path = "signal_class/Videos/Treadmill_Running_500fps.npy"
  ``` 
  
