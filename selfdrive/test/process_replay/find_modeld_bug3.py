#!/usr/bin/env python3
import os
import os
os.environ['SEND_RAW_PRED'] = '1'

import sys
import time
from collections import defaultdict
from typing import Any
from itertools import zip_longest
from tqdm import tqdm
import numpy as np

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.spinner import Spinner
from common.timeout import Timeout
from common.transformations.camera import tici_f_frame_size, tici_d_frame_size
from system.hardware import PC
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.test_processes import format_diff
from system.version import get_commit
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

TEST_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36"
SEGMENT = 0
MAX_FRAMES = 600

VIPC_STREAM = {"roadCameraState": VisionStreamType.VISION_STREAM_ROAD, "driverCameraState": VisionStreamType.VISION_STREAM_DRIVER,
               "wideRoadCameraState": VisionStreamType.VISION_STREAM_WIDE_ROAD}
def get_log_fn(ref_commit, test_route):
  return f"{test_route}_model_tici_{ref_commit}.bz2"


def replace_calib(msg, calib):
  msg = msg.as_builder()
  if calib is not None:
    msg.liveCalibration.rpyCalib = calib.tolist()
  return msg


def model_replay(lr):
  if not PC:
    spinner = Spinner()
    spinner.update("starting model replay")
  else:
    spinner = None

  vipc_server = VisionIpcServer("camerad")
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, *(tici_d_frame_size))
  vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, *(tici_f_frame_size))
  vipc_server.start_listener()

  sm = messaging.SubMaster(['modelV2', 'driverStateV2'])
  pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'driverCameraState', 'liveCalibration', 'lateralPlan'])

  managed_processes['modeld'].start()
  time.sleep(5)
  sm.update(1000)

  log_msgs = []
  last_desire = None
  recv_cnt = defaultdict(int)
  frame_idxs = defaultdict(int)

  # init modeld with valid calibration
  cal_msgs = [msg for msg in lr if msg.which() == "liveCalibration"]
  for _ in range(5):
    pm.send(cal_msgs[0].which(), cal_msgs[0].as_builder())
    time.sleep(0.1)

  msgs = defaultdict(list)
  for msg in lr:
    msgs[msg.which()].append(msg)

  img = None
  cam_msgs = list(zip_longest(msgs['roadCameraState'], msgs['wideRoadCameraState']))[0]
  
  i = 0
  img = np.zeros(3493536, dtype=np.uint8)
  #while True:
  for _ in tqdm(range(600)):
  #for cam_msgs in zip_longest(msgs['roadCameraState'], msgs['wideRoadCameraState']):
    # need a pair of road/wide msgs
    if None in (cam_msgs[0], cam_msgs[1]):
      break

    for msg in cam_msgs:
      if msg is None:
        continue

      if msg.which() in VIPC_STREAM:
        msg = msg.as_builder()
        camera_state = getattr(msg, msg.which())
        
        #img = frs[msg.which()].get(frame_idxs[msg.which()], pix_fmt="nv12")[0]
        frame_idxs[msg.which()] += 1

        # send camera state and frame
        camera_state.frameId = frame_idxs[msg.which()]
        camera_state.timestampSof = camera_state.timestampSof + int(i * 0.05 * 1e9)
        camera_state.timestampEof = camera_state.timestampEof + int(i * 0.05 * 1e9)
        #print(msg)
        pm.send(msg.which(), msg)
        vipc_server.send(VIPC_STREAM[msg.which()], img.flatten().tobytes(), camera_state.frameId,
                          camera_state.timestampSof, camera_state.timestampEof)

        recv = None
        if min(frame_idxs['roadCameraState'], frame_idxs['wideRoadCameraState']) > recv_cnt['modelV2']:
          recv = "modelV2"
        #print(frame_idxs['roadCameraState'], frame_idxs['wideRoadCameraState'], recv_cnt['modelV2'])

        
        # wait for a response
        with Timeout(15, f"timed out waiting for {recv}"):
          if recv:
            recv_cnt[recv] += 1
            log_msgs.append(messaging.recv_one(sm.sock[recv]))
            #print(log_msgs[-1])
      
    if min(frame_idxs['roadCameraState'], frame_idxs['wideRoadCameraState']) > MAX_FRAMES:
      break
    i += 1
  managed_processes['modeld'].stop()


  return log_msgs


if __name__ == "__main__":

  update = "--update" in sys.argv
  replay_dir = os.path.dirname(os.path.abspath(__file__))
  ref_commit_fn = os.path.join(replay_dir, "model_replay_ref_commit")

  # load logs
  lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT)))

  # run replay
  while True:
    log_msgs = model_replay(lr)
