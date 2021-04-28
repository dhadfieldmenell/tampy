from policy_hooks.human_labels.video_renderer import *

rend = VideoRenderer()
rend.wait()
time.sleep(1)
rend.cont()

rend.wait_for_user()

